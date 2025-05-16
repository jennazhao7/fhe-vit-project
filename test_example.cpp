// Combined FHE and Plaintext inference in one file

#include <openfhe.h>
#include "fhe_linear.h"
#include "weight_loader.h"
#include "plaintext_utils.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace lbcrypto;
using namespace plaintext_utils;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./test_example <start_idx> <end_idx>\n";
        return 1;
    }

    int start_idx = std::stoi(argv[1]);
    int end_idx = std::stoi(argv[2]);

    std::ostream& fout = std::cout;

    
    
    std::ifstream label_file("../input/labels.txt");
    std::vector<int> ground_truth_labels;
    int val;
    while (label_file >> val) ground_truth_labels.push_back(val);
    
    
    const int num_samples= ground_truth_labels.size();
    if (ground_truth_labels.empty()) {
        std::cerr << "Error: No labels found in labels.txt\n";
        return 1;
    }
    int correct_plain = 0;
    int correct_fhe = 0;
    int local_samples = 0;


    const size_t PATCH_IN = 768;
    const size_t PATCH_OUT = 192;
    const size_t HIDDEN = 768;
    const size_t CLS = 192;
    //I know the number of classes is 200 for TinyImageNet, but this is specific to the dataset I have
        //this is external knowledge
        //this var diff from num_samples
    const size_t NUM_CLASSES = 200;

    // // === CryptoContext setup ===
    // CCParams<CryptoContextCKKSRNS> params;
    // params.SetSecretKeyDist(SPARSE_TERNARY);
    // params.SetRingDim(1 << 15);
    // params.SetScalingModSize(46);
    // params.SetFirstModSize(50);
    // params.SetMultiplicativeDepth(6);
    // CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    // cc->Enable(PKE);
    // cc->Enable(KEYSWITCH);
    // cc->Enable(LEVELEDSHE);
    // cc->Enable(ADVANCEDSHE);
    // auto keyPair = cc->KeyGen();
    // cc->EvalMultKeyGen(keyPair.secretKey);\
    // cc->EvalSumKeyGen(keyPair.secretKey);
    
    // // const size_t out_dim = 192;
    // // const size_t in_dim = 192;
    // usint num_slots = cc->GetRingDimension() / 2;
    // usint num_batch = num_slots / PATCH_IN;


    // // === Generate EvalRotate keys required for FHELinearLayer and EvalSum ===
    // std::vector<int32_t> rotation_indices;
    // for (size_t i = 0; i < out_dim; ++i)
    //     rotation_indices.push_back(-(int(num_batch) * i));
    // for (int i = 1; i < (int)in_dim; i *= 2)
    //     rotation_indices.push_back(i);
    // cc->EvalRotateKeyGen(keyPair.secretKey, rotation_indices);

    
    // std::vector<float> input_raw(in_dim);
    // std::ifstream fin_in("../input/image_2.bin", std::ios::binary);
    // if (!fin_in) {
    //     std::cerr << "Failed to open image_2.bin\n";
    //     return 1;
    // }
    // fin_in.read(reinterpret_cast<char*>(input_raw.data()), input_raw.size() * sizeof(float));
    // fin_in.close();
    // std::vector<double> input_vec(input_raw.begin(), input_raw.end());
    //patch embed weights
    auto patch_weights = LoadWeightMatrixFromBin("../weights/patch_embed_weight.bin", PATCH_OUT, PATCH_IN);
    // ATTENTION
    auto q_weights = LoadWeightMatrixFromBin("../weights/q_proj.bin", CLS, CLS);
    auto k_weights = LoadWeightMatrixFromBin("../weights/k_proj.bin", CLS, CLS);
    auto v_weights = LoadWeightMatrixFromBin("../weights/v_proj.bin", CLS, CLS);
    // Internal attention projection
    auto attn_out_proj_weights = LoadWeightMatrixFromBin("../weights/attn_out_proj.bin", CLS, CLS);


    // MLP
    auto mlp1_weights = LoadWeightMatrixFromBin("../weights/mlp_dense1.bin", HIDDEN, CLS);
    auto mlp2_weights = LoadWeightMatrixFromBin("../weights/mlp_dense2.bin", CLS, HIDDEN);
    // FINAL PROJ (typically output of attention)
    auto final_weights = LoadWeightMatrixFromBin("../weights/classifier.bin", NUM_CLASSES, CLS);

    // auto patch_weights = LoadWeightMatrixFromBin("../weights/patch_embed_weight.bin", out_dim, in_dim);
    // auto q_weights = LoadWeightMatrixFromBin("../weights/q_proj.bin", out_dim, in_dim);
    // auto k_weights = LoadWeightMatrixFromBin("../weights/k_proj.bin", out_dim, in_dim);
    // auto v_weights = LoadWeightMatrixFromBin("../weights/v_proj.bin", out_dim, in_dim);
    // auto mlp1_weights = LoadWeightMatrixFromBin("../weights/mlp_dense1.bin", out_dim, in_dim);
    // auto mlp2_weights = LoadWeightMatrixFromBin("../weights/mlp_dense2.bin", out_dim, in_dim);
    // auto final_weights = LoadWeightMatrixFromBin("../weights/attn_out_proj.bin", out_dim, in_dim);
    for (int sample_idx = start_idx; sample_idx < end_idx && sample_idx < num_samples; ++sample_idx) {
        std::string filename = "../input/image_" + std::to_string(sample_idx) + ".bin";
        std::ifstream fin_in(filename, std::ios::binary);
        if (!fin_in) {
            std::cerr << "Failed to open " << filename << "\n";
            continue;
        }
    
        std::vector<float> input_raw(PATCH_IN);
        fin_in.read(reinterpret_cast<char*>(input_raw.data()), input_raw.size() * sizeof(float));
        fin_in.close();
        std::vector<double> input_vec(input_raw.begin(), input_raw.end());
    
        int ground_truth_label = ground_truth_labels[sample_idx];
        fout << "\n[SAMPLE " << sample_idx << "] Ground Truth Label: " << ground_truth_label << "\n";
    
        // PLAINTEXT inference block here...
        
        fout << "\n[PLAINTEXT INFERENCE]\n";

        // Patch Embedding Layer: [PATCH_OUT x PATCH_IN] * [PATCH_IN x 1] = [PATCH_OUT x 1]
        std::vector<double> patch_embed_out(PATCH_OUT, 0.0);
        for (size_t i = 0; i < PATCH_OUT; ++i)
            for (size_t j = 0; j < PATCH_IN; ++j)
                patch_embed_out[i] += patch_weights[i][j] * input_vec[j];

        // Q, K, V projections: [CLS x PATCH_OUT] * [PATCH_OUT x 1] = [CLS x 1]
        std::vector<double> q(CLS, 0.0), k(CLS, 0.0), v(CLS, 0.0);
        for (size_t i = 0; i < CLS; ++i)
            for (size_t j = 0; j < PATCH_OUT; ++j) {
                q[i] += q_weights[i][j] * patch_embed_out[j];
                k[i] += k_weights[i][j] * patch_embed_out[j];
                v[i] += v_weights[i][j] * patch_embed_out[j];
            }


        // Compute attention scores (elementwise q[i] * k[i])
        std::vector<double> attn_scores(CLS);
        double denom = 0.0;
        for (size_t i = 0; i < CLS; ++i) {
            attn_scores[i] = std::exp(q[i] * k[i] / std::sqrt(static_cast<double>(CLS)));
            denom += attn_scores[i];
        }

        // Normalize scores (softmax)
        for (size_t i = 0; i < CLS; ++i)
            attn_scores[i] /= denom;

        // Apply attention to values
        std::vector<double> attn_out(CLS, 0.0);
        for (size_t i = 0; i < CLS; ++i)
            attn_out[i] = attn_scores[i] * v[i];
        
        // ✅ NEW: Apply output projection after attention
        std::vector<double> attn_proj(CLS, 0.0);
        for (size_t i = 0; i < CLS; ++i)
            for (size_t j = 0; j < CLS; ++j)
                attn_proj[i] += attn_out_proj_weights[i][j] * attn_out[j];

        // ✅ Residual connection
        std::vector<double> post_residual(CLS);
        for (size_t i = 0; i < CLS; ++i)
            post_residual[i] = attn_proj[i] + patch_embed_out[i];


        auto norm = LayerNorm(post_residual);
        std::vector<double> mlp1(HIDDEN, 0.0);
        for (size_t i = 0; i < HIDDEN; ++i)
            for (size_t j = 0; j < CLS; ++j)
                mlp1[i] += mlp1_weights[i][j] * norm[j];

        auto gelu_out = GELU(mlp1);
        fout << "\n[PLAINTEXT INFERENCE WENT PAST MLP1]\n";
        

        std::vector<double> mlp2(CLS, 0.0);
        for (size_t i = 0; i < CLS; ++i)
            for (size_t j = 0; j < HIDDEN; ++j)
                mlp2[i] += mlp2_weights[i][j] * gelu_out[j];
        fout << "\n[PLAINTEXT INFERENCE WENT PAST MLP2]\n";

        std::cout << "[DEBUG] final_weights.size(): " << final_weights.size() << "\n";
        if (!final_weights.empty()) {
            std::cout << "[DEBUG] final_weights[0].size(): " << final_weights[0].size() << "\n";
        }

        std::vector<double> final_logits_plain(NUM_CLASSES, 0.0);
        for (size_t i = 0; i < NUM_CLASSES; ++i)
            for (size_t j = 0; j < CLS; ++j)
                final_logits_plain[i] += final_weights[i][j] * mlp2[j];  // ✅ correct index
        
        //print raw logits
        for (size_t i = 0; i < 10; ++i) fout << final_logits_plain[i] << " ";
       
        // fout << "[PLAINTEXT] Final logits:\n";
        // for (double val : final_logits_plain) fout << val << " ";
        // fout << "\n";

        

        

        std::vector<double> class_logits_pt(final_logits_plain.begin(), final_logits_plain.begin() + NUM_CLASSES);


        int predicted_plain = std::distance(class_logits_pt.begin(), std::max_element(class_logits_pt.begin(), class_logits_pt.end()));
        fout << "[PLAINTEXT] Predicted Class: " << predicted_plain << "\n";
        // Final result: int predicted_plain;
    
        if (predicted_plain == ground_truth_label)
            correct_plain++;
    
        // // FHE inference block here...
      
        // fout << "\n[FHE INFERENCE]\n";
        // auto patch_embed_weights = PackWeightsForFHE(patch_weights, cc, num_batch);
        // Plaintext input_pt = cc->MakeCKKSPackedPlaintext(input_vec);
        // Ciphertext<DCRTPoly> input_ct = cc->Encrypt(keyPair.publicKey, input_pt);
        // Plaintext bias_pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(num_slots, 0.0));
        // auto output_ct = FHELinearLayer(input_ct, patch_embed_weights, bias_pt, cc, keyPair, out_dim, in_dim);

        // auto packed_q = PackWeightsForFHE(q_weights, cc, num_batch);
        // auto packed_k = PackWeightsForFHE(k_weights, cc, num_batch);
        // auto packed_v = PackWeightsForFHE(v_weights, cc, num_batch);

        // auto ct_q = FHELinearLayer(output_ct, packed_q, bias_pt, cc, keyPair, out_dim, in_dim);
        // auto ct_k = FHELinearLayer(output_ct, packed_k, bias_pt, cc, keyPair, out_dim, in_dim);
        // auto ct_v = FHELinearLayer(output_ct, packed_v, bias_pt, cc, keyPair, out_dim, in_dim);

        // auto ct_score = cc->EvalMult(ct_q, ct_k);
        // ct_score = cc->EvalSum(ct_score, in_dim);
        // double scale = 1.0 / std::sqrt(static_cast<double>(in_dim));
        // ct_score = cc->EvalMult(ct_score, scale);

        // Plaintext decrypted;
        // cc->Decrypt(keyPair.secretKey, ct_score, &decrypted);
        // auto score_vals = decrypted->GetRealPackedValue();
        // // fout << "FHE] Attention Score: (truncated to first 20):\n";
        // // for (size_t i = 0; i < 20 && i < score_vals.size(); ++i)
        // //     fout << score_vals[i] << " ";
        // // fout << "\n";
        

        // auto softmax_vals = Softmax(score_vals);
        // Plaintext softmax_pt = cc->MakeCKKSPackedPlaintext(softmax_vals);
        // auto softmax_ct = cc->Encrypt(keyPair.publicKey, softmax_pt);
        // auto attn_ct = cc->EvalMult(softmax_ct, ct_v);
        // attn_ct = cc->EvalSum(attn_ct, in_dim);

        // auto post_residual_ct = cc->EvalAdd(attn_ct, output_ct);
        // Plaintext decrypted_post;
        // cc->Decrypt(keyPair.secretKey, post_residual_ct, &decrypted_post);
        // auto post_vals = decrypted_post->GetRealPackedValue();

        // auto norm_fhe = LayerNorm(post_vals);
        // auto norm_pt = cc->MakeCKKSPackedPlaintext(norm_fhe);
        // auto norm_ct = cc->Encrypt(keyPair.publicKey, norm_pt);

        // auto packed_mlp1 = PackWeightsForFHE(mlp1_weights, cc, num_batch);
        // auto packed_mlp2 = PackWeightsForFHE(mlp2_weights, cc, num_batch);
        // auto packed_final = PackWeightsForFHE(final_weights, cc, num_batch);

        // auto mlp1_ct = FHELinearLayer(norm_ct, packed_mlp1, bias_pt, cc, keyPair, out_dim, in_dim);
        // Plaintext mlp1_de;
        // cc->Decrypt(keyPair.secretKey, mlp1_ct, &mlp1_de);
        // auto mlp1_vals = mlp1_de->GetRealPackedValue();
        // auto gelu_enc = GELU(mlp1_vals);

        // auto gelu_pt = cc->MakeCKKSPackedPlaintext(gelu_enc);
        // auto gelu_ct = cc->Encrypt(keyPair.publicKey, gelu_pt);

        // auto mlp2_ct = FHELinearLayer(gelu_ct, packed_mlp2, bias_pt, cc, keyPair, out_dim, in_dim);
        // auto final_ct = FHELinearLayer(mlp2_ct, packed_final, bias_pt, cc, keyPair, out_dim, in_dim);
        // Plaintext final_de;
        // cc->Decrypt(keyPair.secretKey, final_ct, &final_de);
        // auto final_vals = final_de->GetRealPackedValue();

        // // fout << "[FHE] Final logits (truncated to first 20):\n";
        // // for (size_t i = 0; i < 20 && i < final_vals.size(); ++i)
        // //     fout << final_vals[i] << " ";
        // // fout << "\n";
        // size_t safe_len = std::min(final_vals.size(), (size_t)num_classes);
        // std::vector<double> class_logits(final_vals.begin(), final_vals.begin() + safe_len);

        // int pred_fhe = std::distance(class_logits.begin(), std::max_element(class_logits.begin(), class_logits.end()));
        // fout << "[FHE] Predicted Class: " << pred_fhe << "\n";
        // // Final result: int pred_fhe;
    
        // if (pred_fhe == ground_truth_label)
        //     correct_fhe++;

        ++local_samples;
    }
    
    fout << "\n[SUMMARY] Range [" << start_idx << ", " << end_idx << ")\n";
    fout << "PLAINTEXT Accuracy: " << (100.0 * correct_plain / local_samples) << "%\n";
    // fout << "FHE Accuracy: " << (100.0 * correct_fhe / local_samples) << "%\n";




    return 0;
}
