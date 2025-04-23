import torch
import numpy as np
import os
from transformers import ViTModel

def extract_and_save_tensor(tensor: torch.Tensor, save_path: str):
    """
    Save a 1D or 2D PyTorch tensor to a .bin file in row-major order (for C++).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tensor_np = tensor.detach().cpu().numpy().astype(np.float32)

    # Save binary
    tensor_np.tofile(save_path)
    print(f"[✓] Saved: {save_path} {tensor_np.shape}")

def main():
    model_name = "WinKawaks/vit-tiny-patch16-224"
    model = ViTModel.from_pretrained(model_name)

    W_patch = model.embeddings.patch_embeddings.projection.weight  # (192, 3, 16, 16)
    W_patch = W_patch.view(192, -1)  # Flatten to (192, 768)
    extract_and_save_tensor(W_patch, "weights/patch_embed_weight.bin")

    extract_and_save_tensor(
        model.encoder.layer[0].attention.attention.query.weight,
        "weights/q_proj.bin"
    )

    extract_and_save_tensor(
        model.encoder.layer[0].attention.attention.key.weight,
        "weights/k_proj.bin"
    )

    extract_and_save_tensor(
        model.encoder.layer[0].attention.attention.value.weight,
        "weights/v_proj.bin"
    )

    extract_and_save_tensor(
        model.encoder.layer[0].attention.output.dense.weight,
        "weights/attn_out_proj.bin"
    )

    extract_and_save_tensor(
        model.encoder.layer[0].intermediate.dense.weight,
        "weights/mlp_dense1.bin"
    )

    extract_and_save_tensor(
        model.encoder.layer[0].output.dense.weight,
        "weights/mlp_dense2.bin"
    )

if __name__ == "__main__":
    main()
