import torch
import numpy as np

# Load model and weights
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224", num_labels=200, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# === 🧪 Print shape checks ===
# print("patch_embed:", model.vit.embeddings.patch_embeddings.projection.weight.shape)
# print("q_proj:", model.vit.encoder.layer[0].attention.attention.query.weight.shape)
# print("mlp1:", model.vit.encoder.layer[0].intermediate.dense.weight.shape)
# print(model.classifier.weight.shape)
print("patch_embed:", model.vit.embeddings.patch_embeddings.projection.weight.shape)
print("q_proj:", model.vit.encoder.layer[0].attention.attention.query.weight.shape)
print("k_proj:", model.vit.encoder.layer[0].attention.attention.key.weight.shape)
print("v_proj:", model.vit.encoder.layer[0].attention.attention.value.weight.shape)
print("attn_out_proj:", model.vit.encoder.layer[0].attention.output.dense.weight.shape)
print("mlp1:", model.vit.encoder.layer[0].intermediate.dense.weight.shape)
print("mlp2:", model.vit.encoder.layer[0].output.dense.weight.shape)
print("classifier:", model.classifier.weight.shape)


def save_weight(tensor, filename):
    np_array = tensor.detach().cpu().numpy().astype(np.float32)
    with open(filename, "wb") as f:
        f.write(np_array.tobytes())

# Save selected weights
patch_weight = model.vit.embeddings.patch_embeddings.projection.weight.view(192, -1)
save_weight(patch_weight, "weights/patch_embed_weight.bin")
#save_weight(model.vit.embeddings.patch_embeddings.projection.weight, "weights/patch_embed_weight.bin")
save_weight(model.vit.encoder.layer[0].attention.attention.query.weight, "weights/q_proj.bin")
save_weight(model.vit.encoder.layer[0].attention.attention.key.weight, "weights/k_proj.bin")
save_weight(model.vit.encoder.layer[0].attention.attention.value.weight, "weights/v_proj.bin")
save_weight(model.vit.encoder.layer[0].attention.output.dense.weight, "weights/attn_out_proj.bin")

# save_weight(model.vit.encoder.layer[0].attention.output.dense.weight, "weights/attn_out_proj.bin")
save_weight(model.vit.encoder.layer[0].intermediate.dense.weight, "weights/mlp_dense1.bin")  # [768, 192]


#save_weight(model.vit.encoder.layer[0].intermediate.dense.weight, "weights/mlp_dense1.bin")
save_weight(model.vit.encoder.layer[0].output.dense.weight, "weights/mlp_dense2.bin")         # [192, 768]

#save_weight(model.vit.encoder.layer[0].output.dense.weight, "weights/mlp_dense2.bin")
save_weight(model.classifier.weight, "weights/classifier.bin")

