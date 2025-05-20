// inference/weight_loader.cpp

#include "weight_loader.h"
#include <fstream>
#include <stdexcept>

std::vector<std::vector<double>> LoadWeightMatrixFromBin(const std::string& path, size_t out_dim, size_t in_dim) {
    std::vector<float> raw(out_dim * in_dim);
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open weight file: " + path);
    fin.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float));
    fin.close();

    std::vector<std::vector<double>> weights(out_dim, std::vector<double>(in_dim));
    for (size_t i = 0; i < out_dim; ++i)
        for (size_t j = 0; j < in_dim; ++j)
            weights[i][j] = raw[i * in_dim + j];

    return weights;
}

std::vector<double> LoadWeightVectorFromBin(const std::string& path, size_t dim) {
    std::ifstream file(path, std::ios::binary);
    std::vector<double> vec(dim);
    file.read(reinterpret_cast<char*>(vec.data()), sizeof(double) * dim);
    return vec;
}