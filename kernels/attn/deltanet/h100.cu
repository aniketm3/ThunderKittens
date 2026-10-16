#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <tuple>
#include <iostream>

#ifdef TORCH_COMPILE
#define TK_COMPILE_LIN_ATTN
#endif

//H100 optimized parameters
//D=16 => NUM_WORKERS 16 ACTIVE_TILES 16 is ok
//D=64 => NUM_WORKERS 12 ACTIVE_TILES 8 is ok
//D=128 => NUM_WORKERS 4 ACTIVE_TILES 2 is ok

// ROWS * ACTIVE_TILES = CHUNK_SIZE
// CHUNK_SIZE * NUM_BLOCKS = N

// We process N tokens by dividing into CHUNKS of size CHUNK_SIZE
// Each block is one CHUNK

// Within a CHUNK:
//   - We have ACTIVE_TILES sub-chunks (aka tiles), each of shape [ROWS x D]
//   - Each tile is cooperatively handled by a warp or thread group (not by a single thread)
//   - Threads load Q/K/V into shared memory tiles for compute

// H100 optimized configuration - more workers and tiles for better utilization
#define NUM_WORKERS 16 // Increased from 8 to 16 for H100
#define ACTIVE_TILES 16 // Increased from 8 to 16 for H100
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

template <int C>
__device__ inline void forwardSubstitutionTile(rt_fl<C, C>& tile) {
    for (int i = 1; i < C; ++i) {
        for (int j = 0; j < i; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < C; ++k) {
                float Tik = reinterpret_cast<float*>(&tile)[i * C + k];  // row-major
                float Tkj = reinterpret_cast<float*>(&tile)[k * C + j];  // row-major
                acc += Tik * Tkj;
            }
            reinterpret_cast<float*>(&tile)[i * C + j] += acc;
        }
    }
}

struct fwd_globals {
    using q_tile = st_bf<ROWS, ATTN_D>;
    using k_tile = st_bf<ROWS, ATTN_D>;
    using v_tile = st_bf<ROWS, ATTN_D>;
    using o_tile = st_bf<ROWS, ATTN_D>;

    // global layouts
    using q_gl     = gl<bf16,  -1, -1, -1, ATTN_D, q_tile>;
    using k_gl     = gl<bf16,  -1, -1, -1, ATTN_D, k_tile>;
    using v_gl     = gl<bf16,  -1, -1, -1, ATTN_D, v_tile>;
    using o_gl     = gl<bf16,  -1, -1, -1, ATTN_D, o_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;

    long unsigned int n;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void delta_attention_fwd(const __grid_constant__ fwd_globals g) {
    // matches how grid is initialized in harness (heads, batches)
    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    int warpid = kittens::warpid(); 
    
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    // H100 optimized shared memory allocation - larger tiles for better utilization
    st_bf<ROWS, ATTN_D> (&qo_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    st_bf<ROWS, ATTN_D> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    st_bf<ROWS, ATTN_D> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    st_bf<ATTN_D, ATTN_D> (&s_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES + 1>();

    int total_block_idx = 0;

    if (warpid < ACTIVE_TILES + 1) {
        zero(s_s[warpid]);
    }
    __syncthreads();

    int n_chunks = g.n / (ACTIVE_TILES * ROWS); // number of chunks we will loop over

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        // Load q, k, and v tiles in BF16
        rt_bf<ROWS, ATTN_D> q, k, v, v_fl, o; // TILE SIZED

        // Memory state (s_state)
        rt_fl<ATTN_D, ATTN_D> s_state;     
        rt_bf<ATTN_D, ATTN_D> s_state_bf;
        rt_fl<ATTN_D, ATTN_D> s_new;
        rt_bf<ATTN_D, ATTN_D> s_new_bf;

        // Intermediate computation tiles
        rt_bf<ROWS, ATTN_D> k_beta, v_beta, W_bf, U_bf, u_bf;
        rt_fl<ROWS, ATTN_D> W, U_fl, u, W_S, o_inter, o_intra, o_fl;
        rt_fl<ROWS, ROWS> T, T_tri, T_tri_row, T_tri_col, T_tri_partial, A, A_tri, identity;
        rt_bf<ROWS, ROWS> T_bf, T_tri_row_bf, T_tri_col_bf, A_bf;
        rt_bf<ATTN_D, ROWS> k_transposed;

        int cur_idx;
        if (warpid < ACTIVE_TILES) {
            cur_idx = chunk * ACTIVE_TILES + warpid;
            load(qo_s[warpid], g.q, {batch, head, cur_idx, 0});
            load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
            load(v_s[warpid], g.v, {batch, head, cur_idx, 0});
        }
        __syncthreads();

        // Initialize all tiles to zero
        zero(s_state);
        zero(s_state_bf);
        zero(s_new);
        zero(s_new_bf);

        zero(k_beta);
        zero(v_beta);
        zero(W_bf);
        zero(U_bf);
        zero(u_bf);

        zero(W);
        zero(U_fl);
        zero(u);
        zero(W_S);
        zero(o_inter);
        zero(o_intra);
        zero(o_fl);
        zero(o);

        zero(T);
        zero(T_tri);
        zero(A);
        zero(A_tri);

        zero(T_bf);
        zero(A_bf);

        zero(k_transposed);

        if (warpid < ACTIVE_TILES) {
            // load from shared
            load(q, qo_s[warpid]);
            load(k, k_s[warpid]);
            load(v, v_s[warpid]);
            load(s_state, s_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);

            // multiply k and v by beta
            mul(k_beta, k, BETA);
            mul(v_beta, v, BETA);

            __syncthreads(); 

            // lower triangular matrix T (based on the chunked delta alg)
            // the masking logic
            transpose_sep(k_transposed, k);
            auto & k_transposed_col = swap_layout_inplace(k_transposed);
            mma_AB(T, k_beta, k_transposed_col, T);
            mul(T, T, -1);
            tril(T_tri, T, 1, 0.0f);

            __syncthreads(); 

            copy(T_tri_partial, T_tri);

            // Forward substitution for matrix inversion
            for (int i = 0; i < ROWS; ++i) {
                for (int j = 0; j < i; ++j) {
                    // dot product of T[i, :] * [T, : j]
                    upper_fill(T_tri_row, T_tri, i, 0.0f);
                    lower_fill(T_tri_row, T_tri, i + 1, 0.0f);

                    left_fill(T_tri_col, T_tri, j, 0.0f);
                    right_fill(T_tri_col, T_tri, j+1, 0.0f);

                    // add dot product to value at existing spot T[i, j]
                    copy(T_tri_row_bf, T_tri_row);
                    copy(T_tri_col_bf, T_tri_col);
                    auto & T_tri_col_col = swap_layout_inplace(T_tri_col_bf);
                    mma_AB(T_tri_partial, T_tri_row_bf, T_tri_col_col, T_tri_partial);
                }
            }

            copy(T_tri, T_tri_partial);

            // add identity matrix
            one(identity);
            make_causal(identity, identity, 0.0f);
            make_causal_t(identity, identity, 0.0f);
            add(T_tri, T_tri, identity);

            // copy T_tri back into T because that's what we use in all comps
            copy(T, T_tri);

            // compute intermediate W and U
            // ROWS x ROWS * ROWS x ATTN_D = ROWS x ATTN_D
            copy(T_bf, T);
            auto & k_beta_col = swap_layout_inplace(k_beta);
            mma_AB(W, T_bf, k_beta_col, W);

            __syncthreads(); 

            auto & v_beta_col = swap_layout_inplace(v_beta);
            mma_AB(U_fl, T_bf, v_beta_col, U_fl);

            __syncthreads(); 

            // delta rule (core)
            // u = U - W @ S;
            copy(W_bf, W);
            copy(s_state_bf, s_state);
            auto & s_state_bf_col = swap_layout_inplace(s_state_bf);
            mma_AB(W_S, W_bf, s_state_bf_col, W_S);
            sub(u, U_fl, W_S);

            __syncthreads(); 

            // o_inter = q @ S;
            // ROWS x ATTN_D = ROWS x ATTN_D * ATTN_D x ATTN_D
            mma_AB(o_inter, q, s_state_bf_col, o_inter);

            __syncthreads(); 

            // A = (q @ k.T).tril();
            // ROWS x ROWS = ROWS x ATTN_D * ATTN_D * ROWS
            mma_AB(A, q, k_transposed_col, A);
            tril(A_tri, A, 0, 0.0f);
            copy(A, A_tri);

            __syncthreads(); 

            // o_intra = A @ u;
            // ROWS x ATTN_D = ROWS x ROWS * ROWS x ATTN_D
            copy(A_bf, A);
            copy(u_bf, u);
            auto & u_bf_col = swap_layout_inplace(u_bf);
            mma_AB(o_intra, A_bf, u_bf_col, o_intra);

            __syncthreads(); 

            // S_new = S + k.T @ u;
            transpose_sep(k_transposed, k);
            mma_AB(s_new, k_transposed, u_bf_col, s_new);
            add(s_new, s_new, s_state);

            __syncthreads(); 

            // store updated S into shared for next tile
            store(s_s[(chunk + warpid + 1) % (ACTIVE_TILES + 1)], s_new);

            __syncthreads(); 

            add(o_fl, o_intra, o_inter);
            __syncthreads(); 
            copy(o, o_fl);
            store(qo_s[warpid], o);

            __syncthreads(); 
        }

        __syncthreads();
        total_block_idx = (total_block_idx + ACTIVE_TILES) % (ACTIVE_TILES + 1);
        __syncthreads();

        if (warpid < ACTIVE_TILES) {
            store(g.o, qo_s[warpid], {batch, head, cur_idx, 0});
        }
        __syncthreads();
    }
}

fwd_globals fwd_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v,
    bf16 *d_o,
    long unsigned int ATTN_B, long unsigned int ATTN_H, long unsigned int ATTN_N
) {
    // global pointers
    std::cout << "H100 init" << std::endl;
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

    // H100 optimized shared memory - increased from 500000 to 1000000
    unsigned long mem_size = 1000000; // H100 has more shared memory
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

#endif
#include "fwd_harness.impl"
