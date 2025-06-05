#ifndef PLAINTEXT_UTILS_H
#define PLAINTEXT_UTILS_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace plaintext_utils {

inline std::vector<double> Softmax(const std::vector<double>& logits) {
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> exps(logits.size());
    double sum = 0.0;

    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_val);
        sum += exps[i];
    }

    for (size_t i = 0; i < logits.size(); ++i)
        exps[i] /= sum;

    return exps;
}

inline std::vector<double> GELU(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);

    for (size_t i = 0; i < x.size(); ++i) {
        double val = std::clamp(x[i], -30.0, 30.0);  // ← clamp extreme values
        double x3 = val * val * val;
        result[i] = 0.5 * val * (1.0 + std::tanh(sqrt_2_over_pi * (val + 0.044715 * x3)));
    }
    return result;
}

inline void GELU_matrix(std::vector<std::vector<double>>& X) {
    for (auto& row : X) {
        row = GELU(row);  // Overwrite with activated vector
    }
}

/**inline std::vector<double> LayerNorm(const std::vector<double>& x, double epsilon = 1e-6) {
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();

    double sq_sum = 0.0;
    for (double v : x)
        sq_sum += (v - mean) * (v - mean);
    double stddev = std::sqrt(sq_sum / x.size());

    std::vector<double> normed(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        normed[i] = (x[i] - mean) / (stddev + epsilon);

    return normed;
}**/



inline std::vector<double> LayerNormVector(
    const std::vector<double>& x,
    const std::vector<double>& gamma,
    const std::vector<double>& beta,
    double epsilon = 1e-2
) {
    size_t dim = x.size();
    double mean = 0.0, var = 0.0;
    for (double v : x) mean += v;
    mean /= dim;

    for (double v : x) var += (v - mean) * (v - mean);
    var /= dim;

    std::vector<double> out(dim);
    for (size_t i = 0; i < dim; ++i)
        out[i] = gamma[i] * ((x[i] - mean) / std::sqrt(var + epsilon)) + beta[i];

    return out;
}

inline std::vector<std::vector<double>> LayerNorm(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& gamma,
    const std::vector<double>& beta,
    double epsilon = 1e-2
) {
    std::vector<std::vector<double>> output;
    for (const auto& row : X) {
        output.push_back(plaintext_utils::LayerNormVector(row, gamma, beta, epsilon));
    }
    return output;
}

} // namespace plaintext_utils

#endif // PLAINTEXT_UTILS_H