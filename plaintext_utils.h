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
    for (size_t i = 0; i < x.size(); ++i) {
        double val = x[i];
        result[i] = 0.5 * val * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (val + 0.044715 * std::pow(val, 3))));
    }
    return result;
}

inline std::vector<double> LayerNorm(const std::vector<double>& x, double epsilon = 1e-6) {
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();

    double sq_sum = 0.0;
    for (double v : x)
        sq_sum += (v - mean) * (v - mean);
    double stddev = std::sqrt(sq_sum / x.size());

    std::vector<double> normed(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        normed[i] = (x[i] - mean) / (stddev + epsilon);

    return normed;
}

} // namespace plaintext_utils

#endif // PLAINTEXT_UTILS_H