// inference/weight_loader.h

#pragma once
#include <vector>
#include <string>
#include "openfhe.h"

using namespace lbcrypto;

std::vector<std::vector<double>> LoadWeightMatrixFromBin(const std::string& path, size_t out_dim, size_t in_dim);

