import torch
import numpy as np
import os
from transformers import ViTForImageClassification

# Setup
os.makedirs("weights", exist_ok=True)
model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224", num_labels=200, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

def save_weight(tensor, name):
    array = tensor.detach().cpu().numpy().astype(np.float32)
    array.tofile(f"weights/{name}.bin")

# === Embedding Weights ===
save_weight(model.vit.embeddings.cls_token, "cls_token")  # [1, 1, 192]
save_weight(model.vit.embeddings.position_embeddings, "position_embeddings")  # [1, 197, 192]

# Patch embedding (Conv2d → flatten to [192, 768] like patch_size = 16×16×3 = 768)
patch_proj = model.vit.embeddings.patch_embeddings.projection.weight.view(192, -1)
save_weight(patch_proj, "patch_embed_weight")
save_weight(model.vit.embeddings.patch_embeddings.projection.bias, "patch_embed_bias")

# === Transformer Blocks ===
for i, layer in enumerate(model.vit.encoder.layer):
    prefix = f"layer_{i}"

    # LayerNorm before attention
    save_weight(layer.layernorm_before.weight, f"{prefix}_ln1_weight")
    save_weight(layer.layernorm_before.bias, f"{prefix}_ln1_bias")

    # Self Attention: Q, K, V
    save_weight(layer.attention.attention.query.weight, f"{prefix}_q_weight")
    save_weight(layer.attention.attention.query.bias, f"{prefix}_q_bias")
    save_weight(layer.attention.attention.key.weight, f"{prefix}_k_weight")
    save_weight(layer.attention.attention.key.bias, f"{prefix}_k_bias")
    save_weight(layer.attention.attention.value.weight, f"{prefix}_v_weight")
    save_weight(layer.attention.attention.value.bias, f"{prefix}_v_bias")

    # Attention output
    save_weight(layer.attention.output.dense.weight, f"{prefix}_attn_proj_weight")
    save_weight(layer.attention.output.dense.bias, f"{prefix}_attn_proj_bias")

    # LayerNorm after attention
    save_weight(layer.layernorm_after.weight, f"{prefix}_ln2_weight")
    save_weight(layer.layernorm_after.bias, f"{prefix}_ln2_bias")

    # MLP
    save_weight(layer.intermediate.dense.weight, f"{prefix}_mlp_fc1_weight")  # [768, 192]
    save_weight(layer.intermediate.dense.bias, f"{prefix}_mlp_fc1_bias")
    save_weight(layer.output.dense.weight, f"{prefix}_mlp_fc2_weight")        # [192, 768]
    save_weight(layer.output.dense.bias, f"{prefix}_mlp_fc2_bias")

# === Final LayerNorm ===
save_weight(model.vit.layernorm.weight, "final_ln_weight")
save_weight(model.vit.layernorm.bias, "final_ln_bias")

# === Classifier ===
save_weight(model.classifier.weight, "classifier_weight")  # [200, 192]
save_weight(model.classifier.bias, "classifier_bias")
