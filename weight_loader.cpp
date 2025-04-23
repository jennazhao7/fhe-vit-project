// inference/weight_loader.cpp

#include "weight_loader.h"
#include <fstream>
#include <stdexcept>

std::vector<std::vector<double>> LoadWeightMatrixFromBin(const std::string& path, size_t out_dim, size_t in_dim) {
    std::vector<std::vector<double>> weights(out_dim, std::vector<double>(in_dim));
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Cannot open weight file: " + path);
    }

    for (size_t i = 0; i < out_dim; ++i) {
        float buffer[in_dim];
        fread(buffer, sizeof(float), in_dim, file);
        for (size_t j = 0; j < in_dim; ++j) {
            weights[i][j] = static_cast<double>(buffer[j]);
        }
    }

    fclose(file);
    return weights;
}

