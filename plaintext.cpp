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

    const size_t PATCH_IN = 768;
    const size_t PATCH_OUT = 192;
    const size_t HIDDEN = 768;
    const size_t CLS = 192;
    const size_t NUM_CLASSES = 200;

    auto patch_weights = LoadWeightMatrixFromBin("../weights/patch_embed_weight.bin", PATCH_OUT, PATCH_IN);
    auto patch_bias = LoadWeightVectorFromBin("../weights/patch_embed_bias.bin", PATCH_OUT);

    auto final_ln_weight = LoadWeightVectorFromBin("../weights/final_ln_weight.bin", CLS);
    auto final_ln_bias = LoadWeightVectorFromBin("../weights/final_ln_bias.bin", CLS);
    auto classifier = LoadWeightMatrixFromBin("../weights/classifier_weight.bin", NUM_CLASSES, CLS);
    auto classifier_bias = LoadWeightVectorFromBin("../weights/classifier_bias.bin", NUM_CLASSES);

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

        fout << "\n[PLAINTEXT INFERENCE]\n";

        // Patch embedding
        std::vector<double> patch_embed_out(PATCH_OUT, 0.0);
        for (size_t i = 0; i < PATCH_OUT; ++i) {
            for (size_t j = 0; j < PATCH_IN; ++j)
                patch_embed_out[i] += patch_weights[i][j] * input_vec[j];
            patch_embed_out[i] += patch_bias[i];
        }

        // Create sequence with 1 CLS token + 196 patch tokens (we use patch_embed_out as 1 token for demo)
        std::vector<std::vector<double>> sequence(197, patch_embed_out);

        // Apply 12 transformer blocks
        for (int l = 0; l < 12; ++l) {
            auto prefix = "../weights/layer_" + std::to_string(l) + "_";
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
        }

        // Final layer norm
        auto normed = LayerNormVector(sequence[0], final_ln_weight, final_ln_bias);  // CLS token only

        std::vector<double> logits(NUM_CLASSES, 0.0);
        for (size_t i = 0; i < NUM_CLASSES; ++i)
            for (size_t j = 0; j < CLS; ++j)
                logits[i] += classifier[i][j] * normed[j];
        for (size_t i = 0; i < NUM_CLASSES; ++i)
            logits[i] += classifier_bias[i];

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