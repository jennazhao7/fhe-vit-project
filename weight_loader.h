// inference/weight_loader.h

#pragma once
#include <vector>
#include <string>
#include "openfhe.h"

using namespace lbcrypto;

std::vector<std::vector<double>> LoadWeightMatrixFromBin(const std::string& path, size_t out_dim, size_t in_dim);
std::vector<double> LoadWeightVectorFromBin(const std::string& path, size_t dim);

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
);