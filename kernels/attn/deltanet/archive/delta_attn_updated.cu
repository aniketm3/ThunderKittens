#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <tuple>
#include <iostream>

#ifdef TORCH_COMPILE
#define TK_COMPILE_LIN_ATTN
#endif

//RTX4090
//D=16 => NUM_WORKERS 16 ACTIVE_TILES 8 is ok
//D=64 => NUM_WORKERS 8 ACTIVE_TILES 4 is ok
//D=128 => NUM_WORKERS 2 ACTIVE_TILES 1 is ok

#define NUM_WORKERS 8 //16 // TODO: do 8 warpid's
#define ACTIVE_TILES 8 //8
#define NUM_THREADS NUM_WORKERS*kittens::WARP_THREADS

#define ROWS 16

#undef ATTN_D
#define ATTN_D 16

#define BETA 0.01f //defing the beta weighting the delta update

using namespace kittens;

// do a cumsum on a tile, starting from some given position total_block_idx
// with [2, 1, 8, 3] and total_block_idx = 2, it will give [13, 14, 8, 11] (loops back)
template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void cumsum_inplace(ST (&x)[N_TILES], int total_block_idx) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;

    for(int i = 1; i < N_TILES; i++) {
        #pragma unroll
        for(int j = threadIdx.x; j < ST::num_elements; j+=STRIDE) {
            x[(total_block_idx+i)%N_TILES].data[j] += x[(total_block_idx+i-1)%N_TILES].data[j];
        }
    }
}

template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void revcumsum_inplace(ST (&x)[N_TILES], int total_block_idx) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;

    for(int i = N_TILES-1; i > 0; i--) {
        #pragma unroll
        for(int j = threadIdx.x; j < ST::num_elements; j+=STRIDE) {
            x[(total_block_idx+i)%N_TILES].data[j] += x[(total_block_idx+i+1)%N_TILES].data[j];
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------------- Forward kernel ------------------------------------------
// ---------------------------------------------------------------------------------------------------

struct fwd_globals {
    using q_tile = st_bf<ROWS, ATTN_D>;
    using k_tile = st_bf<ROWS, ATTN_D>;
    using v_tile = st_bf<ROWS, ATTN_D>;
    using o_tile = st_bf<ROWS, ATTN_D>;

    // using error_tile = st_bf<ROWS, ATTN_D>;
    // using p_tile = st_bf<ROWS, ATTN_D>;
    // using s_state_tile = st_bf<ATTN_D, ATTN_D>;
    // using delta_state_tile = st_bf<ATTN_D, ATTN_D>;
    // using p_state_tile = st_bf<ATTN_D, ATTN_D>;


    // global layouts
    using q_gl     = gl<bf16,  -1, -1, -1, ATTN_D, q_tile>;
    using k_gl     = gl<bf16,  -1, -1, -1, ATTN_D, k_tile>;
    using v_gl     = gl<bf16,  -1, -1, -1, ATTN_D, v_tile>;
    using o_gl     = gl<bf16,  -1, -1, -1, ATTN_D, o_tile>;

    // using error_tile = st_bf<ROWS, ATTN_D>;
    // using p_tile = st_bf<ROWS, ATTN_D>;
    // using s_state_tile = st_bf<ATTN_D, ATTN_D>;
    // using delta_state_tile = st_bf<ATTN_D, ATTN_D>;
    // using p_state_tile = st_bf<ATTN_D, ATTN_D>;



    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;



    long unsigned int n;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void delta_attention_fwd(const __grid_constant__ fwd_globals g) {
        const int batch = blockIdx.y;
        const int head  = blockIdx.x;

        int warpid = kittens::warpid(); 

        extern __shared__ alignment_dummy __shm[]; 
        shared_allocator al((int*)&__shm[0]);

        // TODO
        st_bf<ROWS, ATTN_D> (&qo_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
        st_bf<ROWS, ATTN_D> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
        st_bf<ROWS, ATTN_D> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
        st_bf<ATTN_D, ATTN_D> (&s_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<ROWS, ROWS>, ACTIVE_TILES + 1>();

        st_bf<ROWS, ATTN_D> (&shared_debug)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); //shared tile for debugging
        st_bf<ATTN_D, ROWS> (&shared_debug_T)[ACTIVE_TILES]   = al.allocate<st_bf<ATTN_D, ROWS>, ACTIVE_TILES>(); //shared tile for debugging
        st_bf<ATTN_D, ATTN_D> (&shared_debug_64)[ACTIVE_TILES]   = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES>(); //shared tile for debugging
        int total_block_idx = 0;

        if (warpid < ACTIVE_TILES + 1) {
            zero(s_s[warpid]);
        }
        //zero(s_s[warpid]);

        __syncthreads();

        
        int n_blocks = g.n / (ACTIVE_TILES * ROWS); // number of chunks we will loop over

        int i = 0;
        for (int block = 0; block < std::min(n_blocks, n_blocks); block++) {
            i += 1;
            // Load q, k, and v tiles in BF16 (global format: 16x64)
            rt_bf<ROWS, ATTN_D> q, k;          // [16 x 64]
            rt_bf<ROWS, ATTN_D> v;             // [16 x 64]
            rt_fl<ROWS, ATTN_D> v_fl;          // [16 x 64] float version for arithmetic
            rt_fl<ROWS, ATTN_D> o;             // [16 x 64] output

            // Memory state (s_state) is 64x64 in float.
            // rt_fl<ATTN_D, ATTN_D> s_state;     // [64 x 64] float
            // rt_bf<ATTN_D, ATTN_D> s_state_bf;   // BF16 copy of s_state, [64 x 64]
            // rt_fl<ATTN_D, ATTN_D> s_new;        // new memory state, [64 x 64] float
            // rt_bf<ATTN_D, ATTN_D> s_new_bf;     // BF16 copy of s_new, [64 x 64]
            rt_fl<ROWS, ROWS> s_state; 
            rt_bf<ROWS, ROWS> s_state_bf;
            rt_fl<ROWS, ROWS> s_new; 
            rt_bf<ROWS, ROWS> s_new_bf;

            // Intermediate computation tiles (all [16x64]) in float
            rt_fl<ROWS, ATTN_D> error;         // error = s_state*k^T - v, [16 x 64]
            rt_fl<ROWS, ATTN_D> beta_error;    // [16 x 64] float
            rt_bf<ROWS, ATTN_D> beta_error_bf; // BF16 version, [16 x 64]
            rt_fl<ROWS, ATTN_D> P;             // [16 x 64] float

            // Outer product delta will be 64x64 in float and BF16.
            rt_fl<ROWS, ROWS> delta;       // [64 x 64] float
            rt_bf<ROWS, ROWS> delta_bf;    // [64 x 64] BF16

            zero(s_state);
            zero(s_new);
            zero(error);
            zero(beta_error);
            zero(P);//one(P); //one(P) gets rid of zeros
            zero(delta);
            //one(q);
            // zero(shared_debug[warpid]);

            int cur_idx;

            if (warpid < ACTIVE_TILES) {
                // todo: set cur_idx to 0 and inspect first tile for q
                cur_idx = block * ACTIVE_TILES + warpid; // 0
                load(qo_s[warpid], g.q, {batch, head, cur_idx, 0});
                load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
                load(v_s[warpid], g.v, {batch, head, cur_idx, 0});

            } else {
                // cur_idx = block * ACTIVE_TILES + warpid - ACTIVE_TILES;
                // load(v_s[warpid - ACTIVE_TILES], g.v, {batch, head, cur_idx, 0});
            }
            __syncthreads();

            // // --- Compute P = k * (s_state)^T ---
            if (warpid < ACTIVE_TILES) {
                // the loads
                load(q, qo_s[warpid]);
                load(k, k_s[warpid]);
                load(v, v_s[warpid]);

                zero(s_state);
                load(s_state, s_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]); // load current memory state

                copy(s_state_bf, s_state);

                //auto & s_state_col = swap_layout_inplace(s_state_bf);
                //mma_AB(P, k, s_state_col, P);
                //TODO
                // QUESTION: does swap layout inplace actually transpose it or does it just swap the layout type?
                auto & k_col = swap_layout_inplace(k);
                mma_AB(P, s_state_bf, k_col, P); // TODO

                copy(error, P);
                copy(v_fl, v);
                sub(error, error, v_fl);

                copy(beta_error, error);
                mul(beta_error, beta_error, BETA);
                copy(beta_error_bf, beta_error);

                //rt_bf<ATTN_D, ROWS> k_transposed;
                //auto & k_transposed = swap_layout_inplace(k);
                // transpose_sep(k_transposed, k);

                zero(delta);
                mma_ABt(delta, beta_error_bf, k, delta); // we use mma_ABt so that it transposes k for us

                copy(delta_bf, delta);
                copy(s_new, s_state);
                sub(s_new, s_new, delta);

                store(s_s[(total_block_idx + warpid + 1) % (ACTIVE_TILES + 1)], s_new); // ??
                copy(s_new_bf, s_new);
                auto & q_col = swap_layout_inplace(q);
                mma_AB(o, s_new_bf, q_col, o); // TODO
                store(qo_s[warpid], o);

                __syncthreads();

                cumsum_inplace<NUM_WORKERS>(s_s, total_block_idx);
                __syncthreads();

                if (warpid < ACTIVE_TILES) {
                    rt_bf<ROWS, ROWS> s;
                    load(q, qo_s[warpid]);
                    load(s, s_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);
                    // mma_ABt(o, q, s, o); // TODO
                    // store(shared_debug_64[warpid], s);
                    store(qo_s[warpid], o);
                }

            total_block_idx = (total_block_idx + ACTIVE_TILES) % (ACTIVE_TILES + 1);
            __syncthreads();


            if (warpid < ACTIVE_TILES) {
                //store(shared_debug[warpid], v);
                store(g.o, qo_s[warpid], {batch, head, cur_idx, 0});
                // store(g.o, v_s[warpid - ACTIVE_TILES], {batch, head, cur_idx, 0});
            }
            __syncthreads();
        }
    }
}

fwd_globals fwd_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v,
    bf16 *d_o,
    long unsigned int ATTN_B, long unsigned int ATTN_H, long unsigned int ATTN_N
) {
    // global pointers
    std::cout << "init" << std::endl;
    using globals = fwd_globals;

    using q_tile     = globals::q_tile;
    using k_tile     = globals::k_tile;
    using v_tile     = globals::v_tile;
    using o_tile     = globals::o_tile;

    // global layouts
    using q_gl     = globals::q_gl;
    using k_gl     = globals::k_gl;
    using v_gl     = globals::v_gl;
    using o_gl     = globals::o_gl;

    q_gl     q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, nullptr};
    k_gl     k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, nullptr};
    v_gl     v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, nullptr};
    o_gl     o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};

    globals g{
        q_arg, k_arg, v_arg, o_arg, ATTN_N
    };
    return g;
}


#ifdef TK_COMPILE_LIN_ATTN
#include "pyutils/torch_helpers.cuh"
#include <iostream>
void dispatch_fwd( 
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    int ATTN_B, int ATTN_H, int ATTN_N
){
    fwd_globals g = fwd_init(
        d_q, d_k, d_v,
        d_o,
        ATTN_B, ATTN_H, ATTN_N
    );

    // launch
    unsigned long mem_size = 500000; // 4090
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(
        delta_attention_fwd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    delta_attention_fwd<<<grid,NUM_THREADS,mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

torch::Tensor delta_attn_forward(
    const torch::Tensor q, 
    const torch::Tensor k,
    const torch::Tensor v
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    int B = q.size(0);
    int H = q.size(1);
    int DV = v.size(3);
    int N  = q.size(2);
    int FD = k.size(3);

    // checks
    TORCH_CHECK(k.size(0) == B, "k batch?");
    TORCH_CHECK(k.size(1) == H, "k heads?");
    TORCH_CHECK(k.size(2) == N, "k length?");

    TORCH_CHECK(v.size(0) == B, "v batch?");
    TORCH_CHECK(v.size(1) == H, "v heads?");
    TORCH_CHECK(v.size(2) == N, "v length?");

    // allocate output
    torch::Tensor out = torch::empty({B, H, N, DV}, v.options());

    // convert to bf16
    c10::BFloat16 *q_bf16 = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_bf16 = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_bf16 = v.data_ptr<c10::BFloat16>();
    
    bf16 *d_q = reinterpret_cast<bf16*>(q_bf16);
    bf16 *d_k = reinterpret_cast<bf16*>(k_bf16);
    bf16 *d_v = reinterpret_cast<bf16*>(v_bf16);
    bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());

    dispatch_fwd(
        d_q, d_k, d_v, d_o,
        B, H, N
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return out;
    cudaDeviceSynchronize();
}


// #else
// #ifdef FWD_HARNESS
// #include "4090_harness_fwd.impl"
// #else
// #include "4090_harness_bwd.impl"
// #endif
#endif
#include "fwd_harness.impl"
