#include <openfhe.h>
#include <vector>

using namespace lbcrypto;

std::vector<Plaintext> PackWeightsForFHE(const std::vector<std::vector<double>>& weights,
                                         CryptoContext<DCRTPoly> cc,
                                         usint num_batch) {
    size_t out_dim = weights.size();
    size_t in_dim = weights[0].size();
    std::vector<Plaintext> packed_diagonals;

    for (size_t diag = 0; diag < out_dim; ++diag) {
        std::vector<double> packed(num_batch * in_dim, 0.0);
        for (size_t j = 0; j < in_dim; ++j) {
            for (size_t b = 0; b < num_batch; ++b) {
                packed[b * in_dim + j] = weights[diag][j];
            }
        }
        Plaintext pt = cc->MakeCKKSPackedPlaintext(packed);
        packed_diagonals.push_back(pt);
    }

    return packed_diagonals;
}

Ciphertext<DCRTPoly> FHELinearLayer(const Ciphertext<DCRTPoly>& input_ct,
                                    const std::vector<Plaintext>& packed_weights,
                                    const Plaintext& bias_pt,
                                    CryptoContext<DCRTPoly> cc,
                                    const KeyPair<DCRTPoly>& keyPair,
                                    size_t out_dim,
                                    size_t in_dim) {
    size_t num_slots = cc->GetRingDimension() / 2;
    size_t num_batch = num_slots / in_dim;

    Ciphertext<DCRTPoly> result;

    for (size_t i = 0; i < out_dim; ++i) {
        auto rotated = cc->EvalRotate(input_ct, -int(num_batch) * i);
        auto mult = cc->EvalMult(rotated, packed_weights[i]);

        result = (i == 0) ? mult : cc->EvalAdd(result, mult);
    }

    return cc->EvalAdd(result, bias_pt);
}

Ciphertext<DCRTPoly> encrypt_plain_vector(const std::vector<double>& vec,
                                          CryptoContext<DCRTPoly> cc,
                                          const PublicKey<DCRTPoly>& pk) {
    Plaintext pt = cc->MakeCKKSPackedPlaintext(vec);
    return cc->Encrypt(pk, pt);
}