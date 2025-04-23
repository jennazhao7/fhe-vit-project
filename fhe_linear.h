#pragma once

#include <openfhe.h>
#include <vector>

using namespace lbcrypto;


std::vector<Plaintext> PackWeightsForFHE(
    const std::vector<std::vector<double>>& weights,
    CryptoContext<DCRTPoly> cc,
    usint num_batch);

Ciphertext<DCRTPoly> FHELinearLayer(
    const Ciphertext<DCRTPoly>& input_ct,
    const std::vector<Plaintext>& packed_weights,
    const Plaintext& bias_pt,
    CryptoContext<DCRTPoly> cc,
    const KeyPair<DCRTPoly>& keyPair,
    size_t out_dim,
    size_t in_dim);

Ciphertext<DCRTPoly> encrypt_plain_vector(
    const std::vector<double>& vec,
    CryptoContext<DCRTPoly> cc,
    const PublicKey<DCRTPoly>& pk);
