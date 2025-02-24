#include <thunderkittens.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace tk;

// Delta attention kernel using ThunderKittens tiles
template<int D>
__global__ void delta_attn_ker(delta_attn_globals<D> g) {
    // Get block indices
    const int block_n = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;

    // Load tiles for Q, K, V
    using q_tile = typename delta_attn_globals<D>::q_tile;
    using k_tile = typename delta_attn_globals<D>::k_tile;
    using v_tile = typename delta_attn_globals<D>::v_tile;
    using o_tile = typename delta_attn_globals<D>::o_tile;
    using l_tile = typename delta_attn_globals<D>::l_tile;

    // Initialize shared memory for tiles
    __shared__ typename q_tile::storage_t q_storage;
    __shared__ typename k_tile::storage_t k_storage;
    __shared__ typename v_tile::storage_t v_storage;
    __shared__ typename o_tile::storage_t o_storage;
    __shared__ typename l_tile::storage_t l_storage;

    // Initialize tiles
    q_tile q_t(&q_storage);
    k_tile k_t(&k_storage);
    v_tile v_t(&v_storage);
    o_tile o_t(&o_storage);
    l_tile l_t(&l_storage);

    // Initialize state
    float state[D] = {0};

    // Process sequence elements in blocks
    for (int t = 0; t < g.q.n; t += q_tile::N) {
        // Load Q, K, V blocks
        g.q.load(q_t, batch, head, block_n, t);
        g.k.load(k_t, batch, head, block_n, t);
        g.v.load(v_t, batch, head, block_n, t);

        // L2 normalize Q, K
        q_t.normalize();
        k_t.normalize();

        // Compute QK
        float qk = q_t.dot(k_t);

        // Update state using delta rule
        for (int d = 0; d < D; d++) {
            float v_val = v_t.get(d);
            state[d] = state[d] + qk * (v_val - state[d]);
        }

        // Store output
        for (int d = 0; d < D; d++) {
            o_t.set(d, state[d]);
        }
        g.o.store(o_t, batch, head, block_n, t);
    }
}

// Globals struct for delta attention
template<int D>
struct delta_attn_globals {
    using q_tile = st_bf<4*16, D>;
    using k_tile = st_bf<4*16, D>;
    using v_tile = st_bf<4*16, D>;
    using o_tile = st_bf<4*16, D>;
    using l_tile = col_vec<st_fl<4*16, D>>;

    gl<bf16, -1, -1, -1, -1, q_tile> q;
    gl<bf16, -1, -1, -1, -1, k_tile> k;
    gl<bf16, -1, -1, -1, -1, v_tile> v;
    gl<bf16, -1, -1, -1, -1, o_tile> o;
    gl<float, -1, -1, -1, -1, l_tile> l;

    delta_attn_globals(
        gl<bf16, -1, -1, -1, -1, q_tile> q_,
        gl<bf16, -1, -1, -1, -1, k_tile> k_,
        gl<bf16, -1, -1, -1, -1, v_tile> v_,
        gl<bf16, -1, -1, -1, -1, o_tile> o_,
        gl<float, -1, -1, -1, -1, l_tile> l_
    ) : q(q_), k(k_), v(v_), o(o_), l(l_) {}
}; 