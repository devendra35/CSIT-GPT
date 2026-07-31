"""
CSIT-GPT Model

Decoder-only Transformer
"""

import torch
import torch.nn as nn


from model.config import GPTConfig
from model.transformer import TransformerBlock


class GPT(nn.Module):

    def __init__(self):

        super().__init__()

        config = GPTConfig()


        # Convert token IDs into vectors

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )


        # Learn token positions

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )


        # Transformer stack

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(config.n_layer)
            ]
        )


        self.layer_norm = nn.LayerNorm(
            config.n_embd
        )


        # Convert vectors back to vocabulary

        self.output_head = nn.Linear(
            config.n_embd,
            config.vocab_size
        )


        self.dropout = nn.Dropout(
            config.dropout
        )



    def forward(self, idx):


        batch, tokens = idx.shape


        positions = torch.arange(
            tokens,
            device=idx.device
        )


        # Embeddings

        token_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(
            positions
        )


        x = token_emb + pos_emb


        x = self.dropout(x)


        # Transformer

        x = self.blocks(x)


        x = self.layer_norm(x)


        logits = self.output_head(x)


        return logits