#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <tuple>
#include <iostream>

// #ifdef TORCH_COMPILE
// #define TK_COMPILE_LIN_ATTN
// #endif

// RTX4090 recommended configs
#define NUM_WORKERS 8
#define ACTIVE_TILES 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

#define ROWS 16
#define ATTN_D 64
#define BETA 0.05f

using namespace kittens;

// ---------------------------------------------------------
// Minimal forward globals
// ---------------------------------------------------------
struct fwd_globals {
    // st_bf<ROWS,ATTN_D> are *tile* types but we won't use them here.
    // using q_tile = st_bf<ROWS, ATTN_D>;
    // using k_tile = st_bf<ROWS, ATTN_D>;
    // using v_tile = st_bf<ROWS, ATTN_D>;
    // using o_tile = st_bf<ROWS, ATTN_D>;

    // // global layouts
    // using q_gl = gl<bf16, -1, -1, -1, ATTN_D, q_tile>;
    // using k_gl = gl<bf16, -1, -1, -1, ATTN_D, k_tile>;
    // using v_gl = gl<bf16, -1, -1, -1, ATTN_D, v_tile>;
    // using o_gl = gl<bf16, -1, -1, -1, ATTN_D, o_tile>;

    // // pointers
    // q_gl q;
    // k_gl k;
    // v_gl v;
    // o_gl o;

    // unsigned long n; // sequence length
};

// ---------------------------------------------------------
// Minimal forward kernel
// ---------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void delta_attention_fwd(fwd_globals g) 
{
    // Just read something from g to confirm it’s valid
    // int batch = blockIdx.y;
    // int head  = blockIdx.x;
    // int index = threadIdx.x;

    // So we do a minimal load from g.q to ensure pointers are valid
    // but do not do any advanced indexing or store
    // We'll do a single bf16 load, but not use it
    // auto val = g.q.data[ (batch*g.q.strideB)
    //                    + (head*g.q.strideH)
    //                    + (0*g.q.strideN)
    //                    + (0*g.q.strideC) ];
    // do nothing with val
}

// ---------------------------------------------------------
// Minimal forward init
// ---------------------------------------------------------
fwd_globals fwd_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v,
    bf16 *d_o,
    unsigned long ATTN_B, unsigned long ATTN_H, unsigned long ATTN_N
) {
    // std::cout << "init (bare-bones)..." << std::endl;

    // using globals = fwd_globals;
    // using q_gl = globals::q_gl;
    // using k_gl = globals::k_gl;
    // using v_gl = globals::v_gl;
    // using o_gl = globals::o_gl;

    // // Keep the last 'nullptr' argument because that’s what the original code used
    // q_gl q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, nullptr};
    // k_gl k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, nullptr};
    // v_gl v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, nullptr};
    // o_gl o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};

    // globals g{q_arg, k_arg, v_arg, o_arg, ATTN_N};
    // return g;
}

// #ifdef TK_COMPILE_LIN_ATTN
// #include "pyutils/torch_helpers.cuh"
// #include <iostream>

// // Minimal dispatch
// void dispatch_fwd(bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
//                   int ATTN_B, int ATTN_H, int ATTN_N)
// {
//     auto g = fwd_init(d_q, d_k, d_v, d_o, ATTN_B, ATTN_H, ATTN_N);
//     unsigned long mem_size = 100000; 
//     cudaDeviceSynchronize();

//     // Set dynamic shared memory size to some safe value
//     cudaFuncSetAttribute(
//         delta_attention_fwd,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         mem_size
//     );

//     dim3 grid(ATTN_H, ATTN_B);
//     delta_attention_fwd<<<grid, NUM_THREADS, mem_size>>>(g);

//     CHECK_CUDA_ERROR(cudaGetLastError());
//     cudaDeviceSynchronize();
// }

// // Torch wrapper
// torch::Tensor delta_attn_forward(
//     const torch::Tensor q, 
//     const torch::Tensor k,
//     const torch::Tensor v
// ) {
//     CHECK_INPUT(q);
//     CHECK_INPUT(k);
//     CHECK_INPUT(v);

//     int B = q.size(0);
//     int H = q.size(1);
//     int DV = v.size(3);
//     int N  = q.size(2);

//     // just create empty output
//     torch::Tensor out = torch::empty({B, H, N, DV}, v.options());

//     bf16 *d_q = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
//     bf16 *d_k = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
//     bf16 *d_v = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
//     bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());

//     dispatch_fwd(d_q, d_k, d_v, d_o, B, H, N);

//     CHECK_CUDA_ERROR(cudaGetLastError());
//     cudaDeviceSynchronize();
//     return out;
// }

// #endif

// // We do not include any backward code
#include "fwd_harness.impl"
