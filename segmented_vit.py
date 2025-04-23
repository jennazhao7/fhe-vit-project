import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from fhe_layers import FHELinear


class SegmentedViTForFHE(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.base_model = ViTForImageClassification.from_pretrained(model_name)
        self.vit = self.base_model.vit
        self.classifier = self.base_model.classifier  # Final linear layer
        self._replace_linear_layers()

    def forward(self, pixel_values):
        """
        Step-by-step ViT forward pass, decomposed for FHE + DP.
        Input: pixel_values of shape (B, 3, 224, 224)
        """
        # 1. Patch + position embedding (Linear + Add)
        embeddings = self.vit.embeddings(pixel_values)  # shape: (B, seq_len, hidden_size)

        # 2. Pass through transformer encoder block by block
        hidden_states = embeddings
        for i, encoder_layer in enumerate(self.vit.encoder.layer):
            hidden_states = self.forward_block(hidden_states, encoder_layer, block_id=i)

        # 3. LayerNorm before classification
        norm_output = self.vit.layernorm(hidden_states)

        # 4. Extract [CLS] token and apply classifier
        cls_token = norm_output[:, 0]  # shape: (B, hidden_size)
        logits = self.classifier(cls_token)  # Linear → logits

        return logits

    def forward_block(self, hidden_states, encoder_layer, block_id=0):
        """
        One transformer encoder block, decomposed into:
        - LayerNorm1
        - Multi-head self-attention (QKV, Attention, Output projection)
        - Residual
        - LayerNorm2
        - MLP (Linear → GELU → Linear)
        - Residual
        """
        # Extract components
        norm1 = encoder_layer.layernorm1
        attention = encoder_layer.attention
        norm2 = encoder_layer.layernorm2
        mlp = encoder_layer.intermediate.dense, encoder_layer.output.dense
        act = encoder_layer.intermediate.intermediate_act_fn

        # 1. Attention
        normed_input = norm1(hidden_states)                  # Non-linear
        attn_output = attention(normed_input)[0]             # Contains QKV & softmax
        attn_output = attn_output + hidden_states            # Residual Add

        # 2. MLP
        normed_attn = norm2(attn_output)                     # Non-linear
        mlp_hidden = mlp[0](normed_attn)                     # Linear
        mlp_activated = act(mlp_hidden)                      # GELU (Non-linear)
        mlp_output = mlp[1](mlp_activated)                   # Linear
        final_output = mlp_output + attn_output              # Residual Add

        return final_output
    def _replace_linear_layers(self):
        def replace_linear(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, name, FHELinear(child.in_features, child.out_features, child.bias is not None))
                else:
                    replace_linear(child)  # Recursively go deeper

    # Replace Linear layers in both vit and classifier
        replace_linear(self.vit)
        replace_linear(self.classifier)