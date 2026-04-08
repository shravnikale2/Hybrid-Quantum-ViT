import torch
import torch.nn as nn

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)          # (B, 128, 8, 8)
        x = x.flatten(2)               # (B, 128, 64)
        x = x.transpose(1, 2)          # (B, 64, 128)
        return x


class SmallViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=128, num_heads=4, depth=2, num_classes=10):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.transformer(x)

        cls_output = x[:, 0]

        return self.mlp_head(cls_output)