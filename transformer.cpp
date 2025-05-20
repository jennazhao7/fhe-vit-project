

#include <vector>
#include <cmath>
#include <cassert>
#include "fhe_linear.h"  // or your matmul helpers

#include "plaintext_utils.h"  // LayerNorm
using namespace plaintext_utils;
// Transformer block function for ViT-Tiny
// x: input [seq_len x hidden_dim] — e.g., 197 x 192
// All weights: shapes must match preloaded ViT weight dimensions

std::vector<std::vector<double>> Add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    assert(A.size() == B.size());
    std::vector<std::vector<double>> result(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        assert(A[i].size() == B[i].size());
        result[i].resize(A[i].size());
        for (size_t j = 0; j < A[i].size(); ++j)
            result[i][j] = A[i][j] + B[i][j];
    }
    return result;
}
std::vector<std::vector<double>> TransformerBlock(
    const std::vector<std::vector<double>>& x,
    const std::vector<std::vector<double>>& q_weight,
    const std::vector<double>& q_bias,
    const std::vector<std::vector<double>>& k_weight,
    const std::vector<double>& k_bias,
    const std::vector<std::vector<double>>& v_weight,
    const std::vector<double>& v_bias,
    const std::vector<std::vector<double>>& attn_out_proj_weight,
    const std::vector<double>& attn_out_proj_bias,
    const std::vector<double>& ln1_weight,
    const std::vector<double>& ln1_bias,
    const std::vector<std::vector<double>>& mlp_fc1_weight,
    const std::vector<double>& mlp_fc1_bias,
    const std::vector<std::vector<double>>& mlp_fc2_weight,
    const std::vector<double>& mlp_fc2_bias,
    const std::vector<double>& ln2_weight,
    const std::vector<double>& ln2_bias
) {
    size_t seq_len = x.size();
    size_t dim = x[0].size();

    // --- LayerNorm 1 ---
    auto x_norm1 = LayerNorm(x, ln1_weight, ln1_bias);

    // --- Q, K, V ---
    auto Q = MatMul(x_norm1, q_weight);  // [seq_len x dim]
    AddBias(Q, q_bias);
    auto K = MatMul(x_norm1, k_weight);
    AddBias(K, k_bias);
    auto V = MatMul(x_norm1, v_weight);
    AddBias(V, v_bias);

    // --- Attention: QK^T softmax V ---
    std::vector<std::vector<double>> attn_scores(seq_len, std::vector<double>(seq_len, 0.0));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            double dot = 0.0;
            for (size_t d = 0; d < dim; ++d)
                dot += Q[i][d] * K[j][d];
            attn_scores[i][j] = dot / std::sqrt((double)dim);
        }
        Softmax(attn_scores[i]);
    }

    // --- Attention output ---
    std::vector<std::vector<double>> attn_out(seq_len, std::vector<double>(dim, 0.0));
    for (size_t i = 0; i < seq_len; ++i)
        for (size_t j = 0; j < seq_len; ++j)
            for (size_t d = 0; d < dim; ++d)
                attn_out[i][d] += attn_scores[i][j] * V[j][d];

    auto attn_proj = MatMul(attn_out, attn_out_proj_weight);
    AddBias(attn_proj, attn_out_proj_bias);

    // Residual connection
    auto x_res1 = Add(x, attn_proj);

    // --- LayerNorm 2 ---
    auto x_norm2 = LayerNorm(x_res1, ln2_weight, ln2_bias);

    // --- MLP ---
    auto mlp_hidden = MatMul(x_norm2, mlp_fc1_weight);
    AddBias(mlp_hidden, mlp_fc1_bias);
    GELU(mlp_hidden);

    auto mlp_out = MatMul(mlp_hidden, mlp_fc2_weight);
    AddBias(mlp_out, mlp_fc2_bias);

    // Residual connection
    return Add(x_res1, mlp_out);
}
