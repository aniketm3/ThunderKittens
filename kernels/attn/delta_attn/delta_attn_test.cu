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

#define ONE 16

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
    using q_tile = st_bf<ATTN_D, ONE>;
    using k_tile = st_bf<ATTN_D, ONE>;
    using v_tile = st_bf<ATTN_D, ONE>;
    using o_tile = st_bf<ATTN_D, ONE>;
    // using q_tile = st_bf<1, ATTN_D>;
    // using k_tile = st_bf<1, ATTN_D>;
    // using v_tile = st_bf<1, ATTN_D>;
    // using o_tile = st_bf<1, ATTN_D>;

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
    st_bf<ATTN_D, ATTN_D> (&s_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES + 1>();

    // st_bf<ROWS, ATTN_D> (&error_s)[ACTIVE_TILES] = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    // st_bf<ROWS, ATTN_D> (&p_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    // st_bf<ATTN_D, ATTN_D> (&s_state_s)[ACTIVE_TILES]   = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES>();
    // st_bf<ATTN_D, ATTN_D> (&delta_s)[ACTIVE_TILES]   = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES>();
    // st_bf<ATTN_D, ATTN_D> (&s_new_s)[ACTIVE_TILES]   = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES>();

    // st_bf<ROWS, ATTN_D> (&shared_debug)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); //shared tile for debugging
    // st_bf<ATTN_D, ROWS> (&shared_debug_T)[ACTIVE_TILES]   = al.allocate<st_bf<ATTN_D, ROWS>, ACTIVE_TILES>(); //shared tile for debugging
    // st_bf<ATTN_D, ATTN_D> (&shared_debug_64)[ACTIVE_TILES]   = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES>(); //shared tile for debugging
    int total_block_idx = 0;

    if (warpid < ACTIVE_TILES + 1) {
        zero(s_s[warpid]);
    }
    //zero(s_s[warpid]);

    __syncthreads();

    
    int n_blocks = g.n / (ACTIVE_TILES * ROWS); // number of chunks we will loop over

    int i = 0;
    for (int block = 0; block < n_blocks; block++) {
        i += 1;
        // Load q, k, and v tiles in BF16 (global format: 16x64)
        rt_bf<ATTN_D, ONE> q, k;          // [16 x 64]
        rt_bf<ATTN_D, ONE> v;             // [16 x 64]
        rt_fl<ATTN_D, ONE> v_fl;          // [16 x 64] float version for arithmetic
        rt_fl<ATTN_D, ONE> o;             // [16 x 64] output

        rt_bf<1, ATTN_D> q_t, k_t;          // [16 x 64]
        rt_bf<1, ATTN_D> v_t;             // [16 x 64]
        //rt_fl<1, ATTN_D> v_fl;          // [16 x 64] float version for arithmetic
        rt_fl<1, ATTN_D> o_t;             // [16 x 64] output


        // Memory state (s_state) is 64x64 in float.
        rt_fl<ATTN_D, ATTN_D> s_state;     // [64 x 64] float
        rt_bf<ATTN_D, ATTN_D> s_state_bf;   // BF16 copy of s_state, [64 x 64]
        rt_fl<ATTN_D, ATTN_D> s_new;        // new memory state, [64 x 64] float
        rt_bf<ATTN_D, ATTN_D> s_new_bf;     // BF16 copy of s_new, [64 x 64]


        // Intermediate computation tiles (all [16x64]) in float
        rt_fl<ATTN_D, ONE> error;         // error = s_state*k^T - v, [16 x 64]
        rt_fl<ATTN_D, ONE> beta_error;    // [16 x 64] float
        rt_bf<ATTN_D, ONE> beta_error_bf; // BF16 version, [16 x 64]
        rt_fl<ATTN_D, ONE> P;             // [16 x 64] float

        // Outer product delta will be 64x64 in float and BF16.
        rt_fl<ATTN_D, ATTN_D> delta;       // [64 x 64] float
        rt_bf<ATTN_D, ATTN_D> delta_bf;    // [64 x 64] BF16

        zero(s_state);
        zero(s_new);
        zero(error);
        zero(beta_error);
        zero(P);//one(P); //one(P) gets rid of zeros
        zero(delta);
        // zero(shared_debug[warpid]);

        int cur_idx;

        if (warpid < ACTIVE_TILES) {
            // todo: set cur_idx to 0 and inspect first tile for q
            cur_idx = block * ACTIVE_TILES + warpid;
            load(qo_s[warpid], g.q, {batch, head, cur_idx, 0}); // ROWS x ATTN_D
            load(k_s[warpid], g.k, {batch, head, cur_idx, 0}); // ROWS x ATTN_D
            load(v_s[warpid], g.v, {batch, head, cur_idx, 0}); // ROWS x ATTN_D

        } else {
            // cur_idx = block * ACTIVE_TILES + warpid - ACTIVE_TILES;
            // load(v_s[warpid - ACTIVE_TILES], g.v, {batch, head, cur_idx, 0});
        }
        __syncthreads();

        // // --- Compute P = k * (s_state)^T ---
        if (warpid < ACTIVE_TILES) {
            
            load(q_t, qo_s[warpid]); // 1 x ATTN_D
            load(k_t, k_s[warpid]); // 1 x ATTN_D
            load(v_t, v_s[warpid]); // 1 x ATTN_D

            transpose_sep(q, q_t); // ATTN_D x 1
            transpose_sep(k, k_t); // ATTN_D x 1
            transpose_sep(v, v_t); // ATTN_D x 1

            zero(s_state); // ATTN_D x ATTN_D
            // check
            load(s_state, s_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]); // load current memory state

            copy(s_state_bf, s_state); // ATTN_D x ATTN_D

            auto & k_col = swap_layout_inplace(k);
            mma_AB(P, s_state_bf, k_col, P);
            // ATTN_D x 1 = // ATTN_D x ATTN_D * // ATTN_D x 1

            copy(error, P); // ATTN_D x 1
            copy(v_fl, v); // ATTN_D x 1
            sub(error, error, v_fl); // ATTN_D x 1

            copy(beta_error, error); // ATTN_D x 1
            mul(beta_error, beta_error, BETA); // ATTN_D x 1
            copy(beta_error_bf, beta_error); // ATTN_D x 1

            zero(delta); // ATTN_D x ATTN_D
            //auto & k_col = swap_layout_inplace(k); // 1 x ATTN_D
            mma_ABt(delta, beta_error_bf, k, delta); // we use mma_ABt so that it transposes k for us
            // ATTN_D x ATTN_D = // ATTN_D x 1 * 1 x ATTN_D

            copy(delta_bf, delta); // ATTN_D x ATTN_D
            copy(s_new, s_state); // ATTN_D x ATTN_D
            sub(s_new, s_new, delta); // ATTN_D x ATTN_D

            store(s_s[(total_block_idx + warpid + 1) % (ACTIVE_TILES + 1)], s_new); // ??
            copy(s_new_bf, s_new);
            auto & q_col = swap_layout_inplace(q);
            mma_AB(o, s_new_bf, q_col, o);
            // ATTN_D x 1 = // ATTN_D x ATTN_D * // ATTN_D x 1
            transpose_sep(o_t, o); // 1 x ATTN_D
            store(qo_s[warpid], o_t); // TODO fix layout for this

            }

            __syncthreads();


            cumsum_inplace<NUM_WORKERS>(s_s, total_block_idx);
            __syncthreads();

            if (warpid < ACTIVE_TILES) {
                rt_bf<ATTN_D, ATTN_D> s;
                load(q, qo_s[warpid]);
                load(s, s_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);
                mma_ABt(o, q, s, o);
                // store(shared_debug_64[warpid], s);
                store(qo_s[warpid], o);
            }

        total_block_idx = (total_block_idx + ACTIVE_TILES) % (ACTIVE_TILES + 1);
        __syncthreads();

        if (warpid < ACTIVE_TILES) {
            //store(shared_debug[warpid], k);
            store(g.o, qo_s[warpid], {batch, head, cur_idx, 0});
            // store(g.o, v_s[warpid - ACTIVE_TILES], {batch, head, cur_idx, 0});
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

// ---------------------------------------------------------------------------------------------------
// ----------------------------------------- Backward kernel -----------------------------------------
// ---------------------------------------------------------------------------------------------------


struct bwd_globals {
    using q_tile = st_bf<ROWS, ATTN_D>;
    using k_tile = st_bf<ROWS, ATTN_D>;
    using v_tile = st_bf<ROWS, ATTN_D>;
    using do_tile = st_bf<ROWS, ATTN_D>;

    // global layouts
    using q_gl     = gl<bf16, -1, -1, -1, ATTN_D, q_tile>;
    using k_gl     = gl<bf16, -1, -1, -1, ATTN_D, k_tile>;
    using v_gl     = gl<bf16, -1, -1, -1, ATTN_D, v_tile>;
    using do_gl    = gl<bf16, -1, -1, -1, ATTN_D, do_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    do_gl d_o;

    q_gl dq;
    k_gl dk;
    v_gl dv;

    long unsigned int n;
};

// __global__ __launch_bounds__(NUM_THREADS, 1)
// void delta_attention_bwd(const __grid_constant__ bwd_globals g) {
    
//     const int batch = blockIdx.y;
//     const int head  = blockIdx.x;

//     int warpid = kittens::warpid(); 

//     extern __shared__ alignment_dummy __shm[]; 
//     shared_allocator al((int*)&__shm[0]);

//     st_bf<ROWS, ATTN_D> (&dodqqdk_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // do,dq for 1st loop, q,dk for 2nd loop
//     st_bf<ROWS, ATTN_D> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // k for 1st and 2nd
//     st_bf<ROWS, ATTN_D> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // v for 1st and 2nd
//     st_bf<ATTN_D, ATTN_D> (&hidden_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES + 1>(); // accumulates hidden states (memory state S_t form forward pass
//     st_bf<ATTN_D, ATTN_D> (&dhidden_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES + 1>(); // hidden state gradients (ds)
//     st_bf<ROWS, ATTN_D> (&dodv_s)[ACTIVE_TILES] = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // do,dv for 2nd

//     int total_block_idx = 0;

//     if (warpid < ACTIVE_TILES + 1) {
//         zero(hidden_s[warpid]); 
//         zero(dhidden_s[warpid]);
//     }
    
//     int n_blocks = g.n / (ACTIVE_TILES * ROWS);

//     // first loop: dq
//     for (int block = n_blocks - 1; block >= 0; block--) { // iterate backwards since S_t+1 impacts S_t
//         rt_bf<ROWS, ATTN_D> d_o, k, v;
//         rt_bf<ATTN_D, ROWS> vt;
//         rt_bf<ROWS, ROWS> local_attn_bf; //less precision long term storage
//         rt_fl<ROWS, ROWS> local_attn;
//         rt_fl<ATTN_D, ATTN_D> d_accum;
//         rt_fl<ROWS, ATTN_D> dq;

//         int cur_idx;

//         // load the data -> first half k, do, second half v 
//         if(warpid < ACTIVE_TILES) {
//             cur_idx = block * ACTIVE_TILES + warpid;
//             load(dodqqdk_s[warpid], g.d_o, {batch, head, cur_idx, 0});
//             load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
//         }
//         else {
//             cur_idx = block * ACTIVE_TILES + warpid - ACTIVE_TILES;
//             load(v_s[warpid - ACTIVE_TILES], g.v, {batch, head, cur_idx, 0});
//         }
//         __syncthreads();

//         if (warpid < ACTIVE_TILES) {
//             load(d_o, dodqqdk_s[warpid]); // using dodqqdk_s as d_o
//             load(v, v_s[warpid]);

//             zero(local_attn);
//             mma_ABt(local_attn, d_o, v, local_attn); // local_attn <- d_o * v^T

//             // calculating the decay factor as a mask
//             rt_bf<ROWS, ROWS> decay;
//             zero(decay);

//             //calculaing the decay exponent as a lower triangular matrix
//             bf16* decay_ptr = reinterpret_cast<bf16*>(&decay);
//             #pragma unroll
//             for (int i = 0; i < ROWS; i++) {
//                 for (int j=0; j<ROWS; j++) {
//                     // decay.data[i*ROWS + j] *= __float2bfloat16(powf(BETA, i-j));
//                     decay_ptr[i * ROWS + j] = __float2bfloat16(powf(BETA, (float)(i - j)));
//                 }
//             }

//             // applying the decay factor to the local attention
//             bf16* local_attn_ptr = reinterpret_cast<bf16*>(&local_attn);
//             #pragma unroll
//             for (int i = 0; i < ROWS; i++) {
//                 for (int j=0; j <ROWS; j++) {
//                     local_attn_ptr[i * ROWS + j] *= decay_ptr[i * ROWS + j];
//                 }
//             }
            
//             //calculate local intra attention (will later accumulate over everything else)
//             zero(dq);
//             copy(local_attn_bf, local_attn);
//             load(k, k_s[warpid]);
//             auto &k_col = swap_layout_inplace(k);

//             mma_AB(dq, local_attn_bf, k_col, dq); // dq <- local_attn * k^T
            
//             //find dS
//             zero(d_accum);
//             mma_AtB(d_accum, v, k_col, d_accum); //or seperatly transpose v

//             //add gradient to the accumulated gradient
//             rt_fl<ATTN_D, ATTN_D> d_accum_loaded;
//             load(d_accum_loaded, d_accum[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);

//             bf16* d_accum_ptr = reinterpret_cast<bf16*>(&d_accum);
//             bf16* d_accum_loaded_ptr = reinterpret_cast<bf16*>(&d_accum_loaded);
//             #pragma unroll
//             for (int i=0; i<ATTN_D; i++) {
//                 for (int j=0; j<ATTN_D; j++) {
//                     d_accum_ptr[i*ATTN_D + j] += d_accum_ptr[i*ATTN_D + j] * BETA + d_accum_loaded_ptr[i*ATTN_D + j];
//                 }
//             }

//             store(dhidden_s[(total_block_idx + warpid + 1) % (ACTIVE_TILES + 1)], d_accum);
//         }

//         __syncthreads();
//         revcumsum_inplace<NUM_WORKERS>(dhidden_s, total_block_idx);
//         __syncthreads();

//         // doing actual dq calculation over the accumulated gradients
//         if (warpid < ACTIVE_TILES) {
//             rt_bf<ATTN_D, ATTN_D> gradient;
//             load(d_o, dodqqdk_s[warpid]);
//             load(gradient, dhidden_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);
//             auto &gradient_col = swap_layout_inplace(gradient);
//             mma_AB(dq, d_o, gradient_col, dq);
//             store(dodqqdk_s[warpid], dq);
//         }

//         total_block_idx = (total_block_idx + ACTIVE_TILES) % (ACTIVE_TILES + 1);
//         __syncthreads();
        
//         if (warpid < ACTIVE_TILES) {
//             store(g.dq, dodqqdk_s[warpid], {batch, head, cur_idx, 0});
//         }
//         __syncthreads();
//     }

//     // second loop: dk, dv
//     total_block_idx = 0;

//     if (warpid < ACTIVE_TILES + 1) {
//         zero(hidden_s[warpid]);
//     }

//     for (int block = 0; block < n_blocks; block++) {
//         rt_bf<ROWS, ATTN_D> d_o, q, k, v;
//         rt_bf<ROWS, ATTN_D, col_l> q_col;
//         rt_bf<ATTN_D, ROWS> qt;
//         rt_bf<ROWS, ROWS> local_attn_bf;
//         rt_fl<ROWS, ROWS> local_attn;
//         rt_fl<ATTN_D, ATTN_D> dhidden_accum; // dS accumulated values
//         rt_fl<ROWS, ATTN_D> dk, dv;

//         int cur_idx;
//         if (warpid < ACTIVE_TILES) {
//             cur_idx = block * ACTIVE_TILES + warpid;
//             load(dodqqdk_s[warpid], g.d_o, {batch, head, cur_idx, 0});
//             load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
//         }
//         else {
//             cur_idx = block * ACTIVE_TILES + warpid - ACTIVE_TILES;
//             load(v_s[warpid - ACTIVE_TILES], g.v, {batch, head, cur_idx, 0});
//             load(dodv_s[warpid - ACTIVE_TILES], g.d_o, {batch, head, cur_idx, 0});
//         }
//         __syncthreads();

//         if (warpid < ACTIVE_TILES) {
//             load(d_o, dodv_s[warpid]);
//             load(v, v_s[warpid]);

//             // dk calculaiton start
//             zero(local_attn);
//             mma_ABt(local_attn, v, d_o, local_attn); // local_attn <- v * d_o^T

//             rt_bf<ROWS, ROWS> decay;
//             zero(decay);

//             #pragma unroll
//             for (int i = 0; i < ROWS; i++) {
//                 for (int j=0; j<=i; j++) {
//                     decay.data[i*ROWS + j] *= powf(BETA, i-j)
//                 }
//             }

//             //applying mask to local delta calculated ealirer
//             #pragma unroll
//             for (int i = 0; i < ROWS; i++) {
//                 for (int j=0; j <ROWS; j++) {
//                     local_attn.data[i * ROWS + j] *= decay.data[i * ROWS + j];
//                 }
//             }

//             copy(local_attn_bf, local_attn);
//             load(q, dodqqdk_s[warpid]);
//             swap_layout(q_col, q);

//             zero(dk);
//             mma_AB(dk, local_attn_bf, q_col, dk); // dk <- local_attn * q^T 

//             // dv calculation start
//             load(k, k_s[warpid]);
//             zero(local_attn);
//             mma_ABt(local_attn, k, q, local_attn); // local_attn <- k * q^T

//             //applying decay
//             #pragma unroll
//             for (int i = 0; i < ROWS; i++) {
//                 for (int j=0; j< ROWS; j++) {
//                     local_attn.data[i * ROWS + j] *= decay.data[i * ROWS + j];
//                 }
//             }

//             copy(local_attn_bf, local_attn);

//             zero(dv);
//             auto &d_o_col = swap_layout_inplace(d_o);
//             mma_AB(dv, local_attn_bf, d_o_col, dv); // dv <- local_attn * d_o^T

//             // calculating S_t (hidden state)
//             transpose_sep(qt, q);
//             zero(dhidden_accum);
//             mma_AB(dhidden_accum, qt, d_o_col, dhidden_accum); // hidden_s <- q^T * d_o

//             //again applying decay factor
//             rt_fl<ATTN_D, ATTN_D> hidden_loaded;
//             load(hidden_loaded, dhidden_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);

//             #pragma unroll
//             for (int i=0; i<ATTN_D; i++) {
//                 for (int j=0; j<ATTN_D; j++) {
//                     dhidden_accum.data[i*ATTN_D + j] += hidden_loaded.data[i*ATTN_D + j] * BETA + dhidden_accum.data[i*ATTN_D + j];
//                 }
//             }
//             store(hidden_s[(total_block_idx + warpid + 1) % (ACTIVE_TILES + 1)], dhidden_accum);
//         }

//         __syncthreads();
//         cumsum_inplace<NUM_WORKERS>(hidden_s, total_block_idx);
//         __syncthreads();

//         // next part of dk calculation
//         if (warpid < ACTIVE_TILES) {
//             rt_bf<ATTN_D, ATTN_D> gradient;
//             load(v, v_s[warpid]);
//             load(gradient, hidden_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);
//             auto &gradient_col = swap_layout_inplace(gradient);
//             mma_ABt(dk, v, gradient_col, dk);
//             store(dodqqdk_s[warpid], dk);
//         }

//         __syncthreads();

//         // next part of dv calculation
//         if (warpid < ACTIVE_TILES) {
//             rt_bf<ATTN_D, ATTN_D> gradient;
//             load(k, k_s[warpid]);
//             load(gradient, hidden_s[(total_block_idx + warpid) % (ACTIVE_TILES + 1)]);
//             auto &gradient_col = swap_layout_inplace(gradient);
//             mma_AB(dv, k, gradient, dv);
//             store(dodv_s[warpid], dv);
//         }

//         total_block_idx = (total_block_idx + ACTIVE_TILES) % (ACTIVE_TILES + 1);
//         __syncthreads();

//         if (warpid < ACTIVE_TILES) {
//             store(g.dk, dodqqdk_s[warpid], {batch, head, cur_idx, 0});
//             store(g.dv, dodv_s[warpid], {batch, head, cur_idx, 0});
//         }
//         __syncthreads();
//     }

// }

bwd_globals bwd_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_do,
    bf16 *d_dq, bf16 *d_dk, bf16 *d_dv,
    long unsigned int ATTN_B, long unsigned int ATTN_H, long unsigned int ATTN_N
) {
    // global pointers

    using globals = bwd_globals;

    using q_tile     = globals::q_tile;
    using k_tile     = globals::k_tile;
    using v_tile     = globals::v_tile;
    using do_tile     = globals::do_tile;

    // global layouts
    globals::q_gl  q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::k_gl  k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::v_gl  v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::do_gl do_arg{d_do, ATTN_B, ATTN_H, ATTN_N, nullptr};

    globals::q_gl dq_arg{d_dq, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::k_gl dk_arg{d_dk, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::v_gl dv_arg{d_dv, ATTN_B, ATTN_H, ATTN_N, nullptr};

    globals g{
        q_arg, k_arg, v_arg, do_arg, dq_arg, dk_arg, dv_arg, ATTN_N
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

void dispatch_bwd(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_do,
    bf16 *d_dq, bf16 *d_dk, bf16 *d_dv,
    int ATTN_B, int ATTN_H, int ATTN_N
){
    bwd_globals g = bwd_init(
        d_q, d_k, d_v, d_do,
        d_dq, d_dk, d_dv,
        ATTN_B, ATTN_H, ATTN_N
    );

    // launch
    unsigned long mem_size = 100000; // 4090
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(
        delta_attention_bwd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    delta_attention_bwd<<grid,NUM_THREADS,mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> delta_attn_backward(
    const torch::Tensor q, 
    const torch::Tensor k,
    const torch::Tensor v,
    const torch::Tensor _do
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(_do);

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
    torch::Tensor out_dq = torch::empty({B, H, N, FD}, q.options());
    torch::Tensor out_dk = torch::empty({B, H, N, FD}, k.options());
    torch::Tensor out_dv = torch::empty({B, H, N, DV}, v.options());

    // convert to bf16
    c10::BFloat16 *q_bf16 = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_bf16 = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_bf16 = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *do_bf16 = _do.data_ptr<c10::BFloat16>();
    
    bf16 *d_q = reinterpret_cast<bf16*>(q_bf16);
    bf16 *d_k = reinterpret_cast<bf16*>(k_bf16);
    bf16 *d_v = reinterpret_cast<bf16*>(v_bf16);
    bf16 *d_do = reinterpret_cast<bf16*>(do_bf16);
    bf16 *d_dq = reinterpret_cast<bf16*>(out_dq.data_ptr<c10::BFloat16>());
    bf16 *d_dk = reinterpret_cast<bf16*>(out_dk.data_ptr<c10::BFloat16>());
    bf16 *d_dv = reinterpret_cast<bf16*>(out_dv.data_ptr<c10::BFloat16>());

    dispatch_bwd(
        d_q, d_k, d_v, d_do,
        d_dq, d_dk, d_dv,
        B, H, N
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return std::make_tuple(out_dq, out_dk, out_dv);
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