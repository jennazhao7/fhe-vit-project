//#include <openfhe.h>
//#include "fhe_linear.h"
#include "weight_loader.h"


#include "plaintext_utils.h"
#include <fstream>
#include <iostream>
#include <vector>
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

    const int num_samples = ground_truth_labels.size();
    if (ground_truth_labels.empty()) {
        std::cerr << "Error: No labels found in labels.txt\n";
        return 1;
    }
    int correct_plain = 0;
    int local_samples = 0;

    const int CHANNELS = 3;
    const int HEIGHT = 224;
    const int WIDTH = 224;
    const int PATCH_SIZE = 16;
    const int PATCH_IN = 768;     // 3 * 16 * 16
    const int PATCH_OUT = 192;
    const int NUM_PATCHES = (HEIGHT / PATCH_SIZE) * (WIDTH / PATCH_SIZE); // 14 * 14 = 196
    const size_t HIDDEN = 768;
    const size_t CLS = 192;
    const size_t NUM_CLASSES = 200;

    // Load model weights
    auto patch_weights = LoadWeightMatrixFromBin("../weights/patch_embed_weight.bin", PATCH_OUT, PATCH_IN);
    auto patch_bias = LoadWeightVectorFromBin("../weights/patch_embed_bias.bin", PATCH_OUT);
    auto cls_token = LoadWeightMatrixFromBin("../weights/cls_token.bin", 1, CLS);
    fout << "[DEBUG] cls_token[0]: " << cls_token[0][0] << ", " << cls_token[0][1] << "\n";
    auto pos_embed_raw = LoadWeightMatrixFromBin("../weights/position_embeddings.bin", 1, 197 * CLS);

    auto final_ln_weight = LoadWeightVectorFromBin("../weights/final_ln_weight.bin", CLS);
    auto final_ln_bias = LoadWeightVectorFromBin("../weights/final_ln_bias.bin", CLS);
    auto classifier = LoadWeightMatrixFromBin("../weights/classifier_weight.bin", NUM_CLASSES, CLS);
    auto classifier_bias = LoadWeightVectorFromBin("../weights/classifier_bias.bin", NUM_CLASSES);
    // fout << "[DEBUG] classifier[118][0]: " << classifier[118][0] << ", classifier_bias[118]: " << classifier_bias[118] << "\n";
    // fout << "[DEBUG] classifier[0][0]: " << classifier[0][0] << ", classifier_bias[0]: " << classifier_bias[0] << "\n";
    // Reshape position embeddings [1][197*192] → [197][192]
    std::vector<std::vector<double>> pos_embed(197, std::vector<double>(CLS));
    for (int i = 0; i < 197; ++i)
        for (int j = 0; j < CLS; ++j)
            pos_embed[i][j] = pos_embed_raw[0][i * CLS + j];

    for (int sample_idx = start_idx; sample_idx < end_idx && sample_idx < num_samples; ++sample_idx) {
        std::string filename = "../input/image_" + std::to_string(sample_idx) + ".bin";
        std::ifstream fin_in(filename, std::ios::binary);
        if (!fin_in) {
            std::cerr << "Failed to open " << filename << "\n";
            continue;
        }

        // === Load image [3 * 224 * 224] ===
        std::vector<float> image(CHANNELS * HEIGHT * WIDTH);
        fin_in.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
        fin_in.close();

        // === Extract patch embeddings [196][192] ===
        std::vector<std::vector<double>> patch_embeddings(NUM_PATCHES, std::vector<double>(PATCH_OUT, 0.0));

        for (int ph = 0; ph < HEIGHT / PATCH_SIZE; ++ph) {
            for (int pw = 0; pw < WIDTH / PATCH_SIZE; ++pw) {
                std::vector<double> patch_flat(PATCH_IN); // [768]
                int patch_idx = ph * (WIDTH / PATCH_SIZE) + pw;

                // Extract patch
                for (int c = 0; c < CHANNELS; ++c) {
                    for (int i = 0; i < PATCH_SIZE; ++i) {
                        for (int j = 0; j < PATCH_SIZE; ++j) {
                            int h = ph * PATCH_SIZE + i;
                            int w = pw * PATCH_SIZE + j;
                            int flat_idx = c * HEIGHT * WIDTH + h * WIDTH + w;
                            patch_flat[c * PATCH_SIZE * PATCH_SIZE + i * PATCH_SIZE + j] = image[flat_idx];
                        }
                    }
                }

                // Apply linear projection to 192-dim
                for (int out = 0; out < PATCH_OUT; ++out) {
                    double val = 0.0;
                    for (int in = 0; in < PATCH_IN; ++in)
                        val += patch_weights[out][in] * patch_flat[in];
                    patch_embeddings[patch_idx][out] = val + patch_bias[out];
                }
            }
        }
        fout << "[DEBUG] patch_embeddings[0][0]: " << patch_embeddings[0][0] << "\n";
        fout << "[DEBUG] patch_embeddings[10][0]: " << patch_embeddings[10][0] << "\n";
        fout << "[DEBUG] patch_embeddings[195][0]: " << patch_embeddings[195][0] << "\n";

        // === Form input sequence [197][192] = [CLS] + [196 patches] ===
        std::vector<std::vector<double>> sequence(197);
        sequence[0] = cls_token[0];  // CLS token
        for (int i = 0; i < NUM_PATCHES; ++i)
            sequence[i + 1] = patch_embeddings[i];

        // === Add positional embeddings ===
        for (int i = 0; i < 197; ++i)
            for (int j = 0; j < CLS; ++j)
                sequence[i][j] += pos_embed[i][j];
        
            
    // for (int sample_idx = start_idx; sample_idx < end_idx && sample_idx < num_samples; ++sample_idx) {
    //     std::string filename = "../input/image_" + std::to_string(sample_idx) + ".bin";
    //     std::ifstream fin_in(filename, std::ios::binary);
    //     if (!fin_in) {
    //         std::cerr << "Failed to open " << filename << "\n";
    //         continue;
    //     }

    //     std::vector<float> input_raw(PATCH_IN);
    //     fin_in.read(reinterpret_cast<char*>(input_raw.data()), input_raw.size() * sizeof(float));
    //     fin_in.close();
    //     std::vector<double> input_vec(input_raw.begin(), input_raw.end());
    //     fout << "input_vec[0]: " << input_vec[0] << ", input_vec[1]: " << input_vec[1] << ", sample_idx: " << sample_idx << "\n";

        int ground_truth_label = ground_truth_labels[sample_idx];
         fout << "\n[SAMPLE " << sample_idx << "] Ground Truth Label: " << ground_truth_label << "\n";

    //     fout << "\n[PLAINTEXT INFERENCE]\n";

    //     // Patch embedding
    //     std::vector<double> patch_embed_out(PATCH_OUT, 0.0);
    //     for (size_t i = 0; i < PATCH_OUT; ++i) {
    //         for (size_t j = 0; j < PATCH_IN; ++j)
    //             patch_embed_out[i] += patch_weights[i][j] * input_vec[j];
    //         patch_embed_out[i] += patch_bias[i];
    //     }

    //     // Create sequence with 1 CLS token + 196 patch tokens (we use patch_embed_out as 1 token for demo)
    //     std::vector<std::vector<double>> sequence(197, patch_embed_out);
        
        // Apply 12 transformer blocks
        for (int l = 0; l < 12; ++l) {
            auto prefix = "../weights/layer_" + std::to_string(l) + "_";
            //fout << "[DEBUG] Before Block " << l << ", sequence[0][0]: " << sequence[0][0] << "\n";
            auto out = TransformerBlock(
                sequence,
                LoadWeightMatrixFromBin(prefix + "q_weight.bin", CLS, CLS),
                LoadWeightVectorFromBin(prefix + "q_bias.bin", CLS),
                LoadWeightMatrixFromBin(prefix + "k_weight.bin", CLS, CLS),
                LoadWeightVectorFromBin(prefix + "k_bias.bin", CLS),
                LoadWeightMatrixFromBin(prefix + "v_weight.bin", CLS, CLS),
                LoadWeightVectorFromBin(prefix + "v_bias.bin", CLS),
                LoadWeightMatrixFromBin(prefix + "attn_proj_weight.bin", CLS, CLS),
                LoadWeightVectorFromBin(prefix + "attn_proj_bias.bin", CLS),
                LoadWeightVectorFromBin(prefix + "ln1_weight.bin", CLS),
                LoadWeightVectorFromBin(prefix + "ln1_bias.bin", CLS),
                LoadWeightMatrixFromBin(prefix + "mlp_fc1_weight.bin", HIDDEN, CLS),
                LoadWeightVectorFromBin(prefix + "mlp_fc1_bias.bin", HIDDEN),
                LoadWeightMatrixFromBin(prefix + "mlp_fc2_weight.bin", CLS, HIDDEN),
                LoadWeightVectorFromBin(prefix + "mlp_fc2_bias.bin", CLS),
                LoadWeightVectorFromBin(prefix + "ln2_weight.bin", CLS),
                LoadWeightVectorFromBin(prefix + "ln2_bias.bin", CLS)
            );
            sequence = out;
            // for (auto& token : sequence)
            //     for (auto& x : token)
            //         x = std::max(std::min(x, 10.0), -10.0);
            fout << "[DEBUG] Block " << l << ": CLS token first 5: ";
            for (int i = 0; i < 5; ++i) fout << sequence[0][i] << " ";
            fout << "\n";
        }

        // Final layer norm
        std::fill(final_ln_weight.begin(), final_ln_weight.end(), 1.0);
        std::fill(final_ln_bias.begin(), final_ln_bias.end(), 0.0);
        auto normed = LayerNormVector(sequence[0], final_ln_weight, final_ln_bias);  // CLS token only
        fout << "[DEBUG] normed[0]: " << normed[0] << ", normed[1]: " << normed[1] << "\n";

        std::vector<double> logits(NUM_CLASSES, 0.0);
        for (size_t i = 0; i < NUM_CLASSES; ++i)
            for (size_t j = 0; j < CLS; ++j)
                logits[i] += classifier[i][j] * normed[j];
        for (size_t i = 0; i < NUM_CLASSES; ++i)
            logits[i] += classifier_bias[i];
        // for (size_t i = 0; i < NUM_CLASSES; ++i)
        //     fout << "logits[" << i << "]: " << logits[i] << "\n";
        int predicted_plain = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        fout << "[PLAINTEXT] Predicted Class: " << predicted_plain << "\n";

        if (predicted_plain == ground_truth_label)
            correct_plain++;

        ++local_samples;
    }

    fout << "\n[SUMMARY] Range [" << start_idx << ", " << end_idx << ")\n";
    fout << "PLAINTEXT Accuracy: " << (100.0 * correct_plain / local_samples) << "%\n";

    return 0;
}