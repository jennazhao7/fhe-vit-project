#include <openfhe.h>
#include "fhe_linear.h"
#include "weight_loader.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "plaintext_utils.h"
using namespace lbcrypto;
using namespace plaintext_utils;

int main() {
    // === 1. Setup CryptoContext ===
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(SPARSE_TERNARY);
    params.SetRingDim(1 << 15);
    params.SetScalingModSize(46);
    params.SetFirstModSize(50);
    params.SetMultiplicativeDepth(6);  // enough for 1 matmul
    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
        // === 2. Load patch_embed weights (assume 768x768 for ViT base) ===
    const size_t out_dim = 192;
    const size_t in_dim = 192;  
    // const size_t in_dim = 32;
    // const size_t out_dim = 32;
    usint num_slots = cc->GetRingDimension() / 2;
    usint num_batch = num_slots / in_dim;
        
    std::vector<int32_t> rotation_indices;

    // 1. Add num_batch * i negative shifts (as you originally had)
    for (size_t i = 0; i < out_dim; ++i) {
        rotation_indices.push_back(-(int(num_batch) * i));
    }
    
    // 2. Add powers-of-two positive rotations for EvalSum
    for (int i = 1; i < (int)in_dim; i *= 2) {
        rotation_indices.push_back(i);
    }
    
    // 3. Generate rotation keys
    cc->EvalRotateKeyGen(keyPair.secretKey, rotation_indices);



    std::vector<float> weight_raw(in_dim * out_dim);
    //std::ifstream fin("weights/patch_embed_32x32.bin", std::ios::binary);
     std::ifstream fin("weights/patch_embed_weight.bin", std::ios::binary);
    fin.read(reinterpret_cast<char*>(weight_raw.data()), weight_raw.size() * sizeof(float));
    fin.close();
    std::vector<std::vector<double>> weights(out_dim, std::vector<double>(in_dim));
    for (size_t i = 0; i < out_dim; ++i)
        for (size_t j = 0; j < in_dim; ++j)
            weights[i][j] = weight_raw[i * in_dim + j];

    // === 3. Pack weights into diagonally encoded plaintexts ===
    
    std::vector<Plaintext> patch_embed_weights = PackWeightsForFHE(weights, cc, num_batch);

    // === 4. Create dummy input patch (1 image patch flattened) ===
    std::vector<double> input_vec(in_dim);
    for (size_t i = 0; i < in_dim; ++i)
        input_vec[i] = (double)i / in_dim;  // Dummy normalized values [0,1)

    Ciphertext<DCRTPoly> input_ct = encrypt_plain_vector(input_vec, cc, keyPair.publicKey);

    // === 5. Run patch embedding ===
    Plaintext bias_pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(num_slots, 0.0));

    Ciphertext<DCRTPoly> output_ct =
    FHELinearLayer(input_ct, patch_embed_weights, bias_pt, cc, keyPair, out_dim, in_dim);

    std::vector<float> q_weight_raw(in_dim * out_dim);
    std::ifstream fin_q("weights/q_proj.bin", std::ios::binary);
    fin_q.read(reinterpret_cast<char*>(q_weight_raw.data()), q_weight_raw.size() * sizeof(float));
    fin_q.close();

    std::vector<std::vector<double>> q_weights(out_dim, std::vector<double>(in_dim));
    for (size_t i = 0; i < out_dim; ++i)
        for (size_t j = 0; j < in_dim; ++j)
            q_weights[i][j] = q_weight_raw[i * in_dim + j];

    std::vector<Plaintext> packed_q_weights = PackWeightsForFHE(q_weights, cc, num_batch);

    std::vector<float> k_weight_raw(in_dim * out_dim);
    std::ifstream fin_k("weights/k_proj.bin", std::ios::binary);
    fin_k.read(reinterpret_cast<char*>(k_weight_raw.data()), k_weight_raw.size() * sizeof(float));
    fin_k.close();
    std::vector<std::vector<double>> k_weights(out_dim, std::vector<double>(in_dim));
    for (size_t i = 0; i < out_dim; ++i)
        for (size_t j = 0; j < in_dim; ++j)
            k_weights[i][j] = k_weight_raw[i * in_dim + j];

    std::vector<Plaintext> packed_k_weights = PackWeightsForFHE(k_weights, cc, num_batch);
    
    std::vector<float> v_weight_raw(in_dim * out_dim);
    std::ifstream fin_v("weights/v_proj.bin", std::ios::binary);
    fin_v.read(reinterpret_cast<char*>(v_weight_raw.data()), v_weight_raw.size() * sizeof(float));
    fin_v.close();

    std::vector<std::vector<double>> v_weights(out_dim, std::vector<double>(in_dim));
    for (size_t i = 0; i < out_dim; ++i)
        for (size_t j = 0; j < in_dim; ++j)
            v_weights[i][j] = v_weight_raw[i * in_dim + j];
    std::vector<Plaintext> packed_v_weights = PackWeightsForFHE(v_weights, cc, num_batch);

    // //=== Q projection ===
    // auto q_weights = LoadWeightMatrixFromBin("weights/q_proj.bin", out_dim, in_dim);
    // auto packed_q = PackWeightsForFHE(q_weights, cc, num_batch);
    // Ciphertext<DCRTPoly> ct_q = FHELinearLayer(output_ct, packed_q, bias_pt, cc, keyPair, out_dim, in_dim);

    // // === K projection ===
    // auto k_weights = LoadWeightMatrixFromBin("weights/k_proj.bin", out_dim, in_dim);
    // auto packed_k = PackWeightsForFHE(k_weights, cc, num_batch);
    // Ciphertext<DCRTPoly> ct_k = FHELinearLayer(output_ct, packed_k, bias_pt, cc, keyPair, out_dim, in_dim);

    // // === V projection ===
    // auto v_weights = LoadWeightMatrixFromBin("weights/v_proj.bin", out_dim, in_dim);
    // auto packed_v = PackWeightsForFHE(v_weights, cc, num_batch);
    // Ciphertext<DCRTPoly> ct_v = FHELinearLayer(output_ct, packed_v, bias_pt, cc, keyPair, out_dim, in_dim);

    Ciphertext<DCRTPoly> ct_q = FHELinearLayer(output_ct, packed_q_weights, bias_pt, cc, keyPair, out_dim, in_dim);
    Ciphertext<DCRTPoly> ct_k = FHELinearLayer(output_ct, packed_k_weights, bias_pt, cc, keyPair, out_dim, in_dim);
    Ciphertext<DCRTPoly> ct_v = FHELinearLayer(output_ct, packed_v_weights, bias_pt, cc, keyPair, out_dim, in_dim);
//     // === Load V projection weights ===
//     std::vector<float> v_weight_raw(in_dim * out_dim);
//     std::ifstream fin_v("weights/v_proj.bin", std::ios::binary);
//     fin_v.read(reinterpret_cast<char*>(v_weight_raw.data()), v_weight_raw.size() * sizeof(float));
//     fin_v.close();

//     std::vector<std::vector<double>> v_weights(out_dim, std::vector<double>(in_dim));
//     for (size_t i = 0; i < out_dim; ++i)
//         for (size_t j = 0; j < in_dim; ++j)
//             v_weights[i][j] = v_weight_raw[i * in_dim + j];

//     std::vector<Plaintext> packed_v_weights = PackWeightsForFHE(v_weights, cc, num_batch);
//     Plaintext v_bias_pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(num_slots, 0.0));
//     Ciphertext<DCRTPoly> ct_k = FHELinearLayer(output_ct, packed_k_weights, k_bias_pt, cc, keyPair, out_dim, in_dim);
//     Ciphertext<DCRTPoly> ct_v = FHELinearLayer(output_ct, packed_v_weights, v_bias_pt, cc, keyPair, out_dim, in_dim);

//     Ciphertext<DCRTPoly> ct_q = FHELinearLayer(output_ct, packed_q_weights, q_bias_pt, cc, keyPair, out_dim, in_dim);
//    // === Q projection ===
   
//    auto packed_q = PackWeightsForFHE(q_weights, cc, num_batch);
//    Ciphertext<DCRTPoly> ct_q = FHELinearLayer(output_ct, packed_q, bias_pt, cc, keyPair, out_dim, in_dim);

//    // === K projection ===
//    auto k_weights = LoadWeightMatrixFromBin("weights/k_proj.bin", out_dim, in_dim);
//    auto packed_k = PackWeightsForFHE(k_weights, cc, num_batch);
//    Ciphertext<DCRTPoly> ct_k = FHELinearLayer(output_ct, packed_k, bias_pt, cc, keyPair, out_dim, in_dim);

//    // === V projection ===
//    auto v_weights = LoadWeightMatrixFromBin("weights/v_proj.bin", out_dim, in_dim);
//    auto packed_v = PackWeightsForFHE(v_weights, cc, num_batch);
//    Ciphertext<DCRTPoly> ct_v = FHELinearLayer(output_ct, packed_v, bias_pt, cc, keyPair, out_dim, in_dim);

   // === Attention score Q * K^T / sqrt(d) ===
   Ciphertext<DCRTPoly> ct_score = cc->EvalMult(ct_q, ct_k);        // Elementwise Q * K
   ct_score = cc->EvalSum(ct_score, in_dim);                        // Sum across dimensions (dot product)
   double scale = 1.0 / std::sqrt(static_cast<double>(in_dim));
   ct_score = cc->EvalMult(ct_score, scale);

    // === 6. Decrypt for debugging ===
    Plaintext result_pt;
    cc->Decrypt(keyPair.secretKey, ct_score, &result_pt);
    auto vals = result_pt->GetRealPackedValue();
    // === apply softmax function on decrypted vals. 
    std::vector<double> softmax_vals = Softmax(vals);


    Plaintext softmax_pt = cc->MakeCKKSPackedPlaintext(softmax_vals);
    Ciphertext<DCRTPoly> softmax_ct = cc->Encrypt(keyPair.publicKey, softmax_pt);

    Ciphertext<DCRTPoly> attn_output = cc->EvalMult(softmax_ct, ct_v);  // Element-wise
    attn_output = cc->EvalSum(attn_output, in_dim);  // Sum for weighted combination

    // === Residual Add ===
    std::cout << "[DEBUG] Applying residual connection..." << std::endl;
    Ciphertext<DCRTPoly> post_residual = cc->EvalAdd(attn_output, output_ct);
    // === LayerNorm (plaintext) ===
    std::cout << "[DEBUG] Decrypting for LayerNorm..." << std::endl;
    Plaintext norm_input_pt;
    cc->Decrypt(keyPair.secretKey, post_residual, &norm_input_pt);
    auto norm_input_vals = norm_input_pt->GetRealPackedValue();

    auto norm_vals = plaintext_utils::LayerNorm(norm_input_vals);
    Plaintext norm_pt = cc->MakeCKKSPackedPlaintext(norm_vals);
    Ciphertext<DCRTPoly> norm_ct = cc->Encrypt(keyPair.publicKey, norm_pt);

    // === MLP Dense 1 ===
    std::cout << "[DEBUG] Running MLP Dense Layer 1..." << std::endl;

    auto mlp1_weights = LoadWeightMatrixFromBin("../weights/mlp_dense1.bin", out_dim, in_dim);
    auto packed_mlp1 = PackWeightsForFHE(mlp1_weights, cc, num_batch);
    auto mlp1_out = FHELinearLayer(norm_ct, packed_mlp1, bias_pt, cc, keyPair, out_dim, in_dim);

    // === GELU (plaintext) ===
    std::cout << "[DEBUG] Decrypting for GELU activation..." << std::endl;
    Plaintext mlp1_pt;
    cc->Decrypt(keyPair.secretKey, mlp1_out, &mlp1_pt);
    auto mlp1_vals = mlp1_pt->GetRealPackedValue();
    auto gelu_vals = GELU(mlp1_vals);
 
    Plaintext gelu_pt = cc->MakeCKKSPackedPlaintext(gelu_vals);
    Ciphertext<DCRTPoly> gelu_ct = cc->Encrypt(keyPair.publicKey, gelu_pt);

    // === MLP Dense 2 ===
    std::cout << "[DEBUG] Running MLP Dense Layer 2..." << std::endl;
    auto mlp2_weights = LoadWeightMatrixFromBin("../weights/mlp_dense2.bin", out_dim, in_dim);
    auto packed_mlp2 = PackWeightsForFHE(mlp2_weights, cc, num_batch);
    auto mlp2_out = FHELinearLayer(gelu_ct, packed_mlp2, bias_pt, cc, keyPair, out_dim, in_dim);

    auto final_weights = LoadWeightMatrixFromBin("../weights/attn_out_proj.bin", out_dim, in_dim);
    auto packed_final = PackWeightsForFHE(final_weights, cc, num_batch);
    Ciphertext<DCRTPoly> final_logits_ct = FHELinearLayer(mlp2_out, packed_final, bias_pt, cc, keyPair, out_dim, in_dim);


    Plaintext final_logits_pt;
    cc->Decrypt(keyPair.secretKey, final_logits_ct, &final_logits_pt);
    auto logits = final_logits_pt->GetRealPackedValue();
    std::cout << "[DEBUG] Final logits (first 10):\n";
    for (size_t i = 0; i < 10; ++i) std::cout << logits[i] << " ";
    std::cout << std::endl;

    auto max_it = std::max_element(logits.begin(), logits.end());
    int predicted_label = std::distance(logits.begin(), max_it);
    std::cout << "[RESULT]Predicted label: " << predicted_label << std::endl;

    return 0;
}
// #include <openfhe.h>
// #include "fhe_linear.h"
// #include "weight_loader.h"
// #include <fstream>
// #include <iostream>
// #include <vector>

// using namespace lbcrypto;

// int main() {
//     // === 1. Setup CryptoContext ===
//     CCParams<CryptoContextCKKSRNS> params;
//     params.SetSecretKeyDist(SPARSE_TERNARY);
//     params.SetRingDim(1 << 15);
//     params.SetScalingModSize(46);
//     params.SetFirstModSize(50);
//     params.SetMultiplicativeDepth(6);  // enough for 1 matmul
//     CryptoContext<DCRTPoly> cc = GenCryptoContext(params);

//     cc->Enable(PKE);
//     cc->Enable(KEYSWITCH);
//     cc->Enable(LEVELEDSHE);

//     KeyPair<DCRTPoly> keyPair = cc->KeyGen();
//     cc->EvalMultKeyGen(keyPair.secretKey);

//     const size_t out_dim = 192;
//     const size_t in_dim = 192;  
//     usint num_slots = cc->GetRingDimension() / 2;
//     usint num_batch = num_slots / in_dim;

//     std::vector<int32_t> rotation_indices;
//     for (size_t i = 0; i < out_dim; ++i) {
//         rotation_indices.push_back(-(int(num_batch) * static_cast<int>(i)));
//     }
//     cc->EvalRotateKeyGen(keyPair.secretKey, rotation_indices, keyPair.publicKey);

//     // === 2. Create dummy input patch (1 image patch flattened) ===
//     std::vector<double> input_vec(in_dim);
//     for (size_t i = 0; i < in_dim; ++i)
//         input_vec[i] = static_cast<double>(i) / in_dim;  // Dummy normalized values [0,1)

//     Ciphertext<DCRTPoly> input_ct = encrypt_plain_vector(input_vec, cc, keyPair.publicKey);
//     Plaintext bias_pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(num_slots, 0.0));

//     // === Patch embedding ===
//     auto patch_weights = LoadWeightMatrixFromBin("weights/patch_embed_weight.bin", out_dim, in_dim);
//     auto packed_patch = PackWeightsForFHE(patch_weights, cc, num_batch);
//     Ciphertext<DCRTPoly> embedded_ct = FHELinearLayer(input_ct, packed_patch, bias_pt, cc, keyPair, out_dim, in_dim);

//     // === Q projection ===
//     auto q_weights = LoadWeightMatrixFromBin("weights/q_proj.bin", out_dim, in_dim);
//     auto packed_q = PackWeightsForFHE(q_weights, cc, num_batch);
//     Ciphertext<DCRTPoly> ct_q = FHELinearLayer(embedded_ct, packed_q, bias_pt, cc, keyPair, out_dim, in_dim);

//     // === K projection ===
//     auto k_weights = LoadWeightMatrixFromBin("weights/k_proj.bin", out_dim, in_dim);
//     auto packed_k = PackWeightsForFHE(k_weights, cc, num_batch);
//     Ciphertext<DCRTPoly> ct_k = FHELinearLayer(embedded_ct, packed_k, bias_pt, cc, keyPair, out_dim, in_dim);

//     // === V projection ===
//     auto v_weights = LoadWeightMatrixFromBin("weights/v_proj.bin", out_dim, in_dim);
//     auto packed_v = PackWeightsForFHE(v_weights, cc, num_batch);
//     Ciphertext<DCRTPoly> ct_v = FHELinearLayer(embedded_ct, packed_v, bias_pt, cc, keyPair, out_dim, in_dim);

//     // === Attention score Q * K^T / sqrt(d) ===
//     Ciphertext<DCRTPoly> ct_score = cc->EvalMult(ct_q, ct_k);        // Elementwise Q * K
//     ct_score = cc->EvalSum(ct_score, in_dim);                        // Sum across dimensions (dot product)
//     double scale = 1.0 / std::sqrt(static_cast<double>(in_dim));
//     ct_score = cc->EvalMult(ct_score, scale);

//     // === Decrypt for debugging ===
//     Plaintext result_pt;
//     cc->Decrypt(keyPair.secretKey, ct_score, &result_pt);
//     auto vals = result_pt->GetRealPackedValue();

//     std::cout << "Attention Score (Q*K^T / sqrt(d)) first 10 values: ";
//     for (size_t i = 0; i < 10; ++i)
//         std::cout << vals[i] << " ";
//     std::cout << std::endl;

//     return 0;
// }
// #include "openfhe.h"
// #include "fhe_linear.h"
// #include "weight_loader.h"

// using namespace lbcrypto;

// int main() {
//     // === 1. Set up CryptoContext ===
//     CCParams<CryptoContextCKKSRNS> params;
//     params.SetSecurityLevel(HEStd_128_classic);
//     params.SetRingDim(1 << 15);
//     params.SetScalingModSize(46);
//     params.SetMultiplicativeDepth(10);

//     CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
//     cc->Enable(PKE);
//     cc->Enable(LEVELEDSHE);
//     cc->Enable(KEYSWITCH);

//     auto keyPair = cc->KeyGen();
//     cc->EvalMultKeyGen(keyPair.secretKey);

//     std::vector<int> rotations;
//     for (int i = 1; i <= 768; i *= 2) rotations.push_back(i), rotations.push_back(-i);
//     cc->EvalRotateKeyGen(keyPair.secretKey, rotations);

//     // === 2. Load patch embedding weights and pack ===
//     size_t in_dim = 768;
//     size_t out_dim = 768;
//     size_t num_slots = cc->GetRingDimension() / 2;
//     size_t num_batch = num_slots / in_dim;

//     auto patch_W = LoadWeightMatrixFromBin("weights/patch_embed_weight.bin", out_dim, in_dim);
//     auto packed_patch_W = PackWeightsForFHE(patch_W, cc, num_batch);

//     std::vector<double> patch_bias(out_dim, 0.0);
//     Plaintext pt_patch_bias = cc->MakeCKKSPackedPlaintext(patch_bias);

//     // === 3. Encrypt dummy input patch ===
//     std::vector<double> input_patch(in_dim, 1.0);
//     Plaintext pt_input_patch = cc->MakeCKKSPackedPlaintext(input_patch);
//     auto ct_patch_embed = cc->Encrypt(keyPair.publicKey, pt_input_patch);

//     // === 4. Run patch embedding (FHELinearLayer) ===
//     auto ct_embedded = FHELinearLayer(ct_patch_embed, packed_patch_W, pt_patch_bias, cc, keyPair, out_dim, in_dim);

//     // === 5. Load Q projection weights and pack ===
//     auto q_W = LoadWeightMatrixFromBin("weights/q_proj.bin", out_dim, in_dim);
//     auto packed_q_W = PackWeightsForFHE(q_W, cc, num_batch);

//     std::vector<double> q_bias(out_dim, 0.0);
//     Plaintext pt_q_bias = cc->MakeCKKSPackedPlaintext(q_bias);

//     // === 6. Run Q projection (FHELinearLayer again) ===
//     auto ct_q = FHELinearLayer(ct_embedded, packed_q_W, pt_q_bias, cc, keyPair, out_dim, in_dim);

//     // === 7. Decrypt and view Q projection output ===
//     Plaintext pt_q_out;
//     cc->Decrypt(keyPair.secretKey, ct_q, &pt_q_out);
//     pt_q_out->SetLength(out_dim);

//     std::cout << "\n[Q Projection Output (First 10 values)]\n";
//     for (int i = 0; i < 10; ++i) {
//         std::cout << pt_q_out->GetRealPackedValue()[i] << " ";
//     }
//     std::cout << "...\n";

//     return 0;
// }