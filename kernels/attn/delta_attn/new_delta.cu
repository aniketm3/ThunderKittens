#include <thunderkittens.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace tk;
using namespace kittens;

template<int D> struct delta_attn_ker_tile_dims {
    constexpr static int tile_width = D;
    constexpr static int tile_h = 4*16;
    constexpr static int blocks_sm = (D == 64) ? 2 : 1;
};

constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;

template<int D>
struct delta_attn_globals {
    using q_tile = st_bf<4*16, D>;
    using k_tile = st_bf<4*16, D>;
    using v_tile = st_bf<4*16, D>;
    using o_tile = st_bf<4*16, D>;
    using l_tile = col_vec<st_fl<4*16, D>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    l_gl l;
    float beta;
    int seq_len;
};

template<int D>
__global__ __launch_bounds__(NUM_WARPGROUPS * WARP_THREADS, (D == 64) ? 2 : 1)
void delta_attn_ker(const __grid_constant__ delta_attn_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid / 4;
    const int seq_idx = blockIdx.x;

    using q_tile = st_bf<4*16, D>;
    using k_tile = st_bf<4*16, D>;
    using v_tile = st_bf<4*16, D>;
    using o_tile = st_bf<4*16, D>;
    using l_tile = col_vec<st_fl<4*16, D>>;

    // Shared memory allocations
    q_tile (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile (&k_smem)[8] = al.allocate<k_tile, 8>();
    v_tile (&v_smem)[8] = al.allocate<v_tile, 8>();
    o_tile (&o_smem)[CONSUMER_WARPGROUPS] = al.allocate<o_tile, CONSUMER_WARPGROUPS>();
    l_tile (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_tile, CONSUMER_WARPGROUPS>();

    // Semaphores for pipeline synchronization
    __shared__ kittens::semaphore qsmem_semaphore;
    __shared__ kittens::semaphore k_smem_arrived[8];
    __shared__ kittens::semaphore v_smem_arrived[8];
    __shared__ kittens::semaphore compute_done[8];

    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for (int i = 0; i < 8; i++) {
            init_semaphore(k_smem_arrived[i], 0, 1);
            init_semaphore(v_smem_arrived[i], 0, 1);
            init_semaphore(compute_done[i], 0, 1);
        }
        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem[0]) * CONSUMER_WARPGROUPS);
    }
    __syncthreads();

    // Producer warp loads data
    if (warpid < PRODUCER_WARPGROUPS) {
        // Load Q tiles
        for (int w = 0; w < CONSUMER_WARPGROUPS; w++) {
            int4 q_tile_idx = {blockIdx.z, blockIdx.y, seq_idx + w, 0};
            tma::load_async(q_smem[w], g.q, q_tile_idx, qsmem_semaphore);
        }

        // Pipeline K,V loads
        const int kv_blocks = (g.seq_len + 7) / 8;
        for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
            int4 kv_tile_idx = {blockIdx.z, blockIdx.y, kv_idx, 0};
            tma::load_async(k_smem[kv_idx%8], g.k, kv_tile_idx, k_smem_arrived[kv_idx%8]);
            tma::load_async(v_smem[kv_idx%8], g.v, kv_tile_idx, v_smem_arrived[kv_idx%8]);
            
            if (kv_idx > 0) {
                wait(compute_done[(kv_idx-1)%8], (kv_idx/8)%2);
            }
        }
    }
    // Consumer warps process data
    else {
        rt_fl<16, D> state_reg;
        zero(state_reg);

        wait(qsmem_semaphore, 0);

        const int kv_blocks = (g.seq_len + 7) / 8;
        for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
            wait(k_smem_arrived[kv_idx%8], (kv_idx/8)%2);
            wait(v_smem_arrived[kv_idx%8], (kv_idx/8)%2);

            rt_fl<16, D> q_reg, k_reg, v_reg;
            load(q_reg, q_smem[warpgroupid-PRODUCER_WARPGROUPS]);
            load(k_reg, k_smem[kv_idx%8]);
            load(v_reg, v_smem[kv_idx%8]);

            // L2 normalize Q,K
            l2_normalize(q_reg);
            l2_normalize(k_reg);

            // Compute attention and update state
            float score = dot(q_reg, k_reg);
            mul(v_reg, v_reg, score * g.beta);
            mul(state_reg, state_reg, 1.0f - (score * g.beta));
            add(state_reg, state_reg, v_reg);

            if (warpgroup::laneid() == 0) {
                arrive(compute_done[kv_idx%8], 1);
            }
        }

        // Store final output
        store(o_smem[warpgroupid-PRODUCER_WARPGROUPS], state_reg);
        warpgroup::sync(warpgroupid);

        if (warpid % 4 == 0) {
            int4 o_tile_idx = {blockIdx.z, blockIdx.y, seq_idx + (warpgroupid-PRODUCER_WARPGROUPS), 0};
            tma::store_async(g.o, o_smem[warpgroupid-PRODUCER_WARPGROUPS], o_tile_idx);
        }
        tma::store_async_wait();
    }
}

