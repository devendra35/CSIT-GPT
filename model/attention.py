


import torch
import torch.nn as nn

from model.config import GPTConfig



class SelfAttention(nn.Module):


    def __init__(self):

        super().__init__()


        self.config = GPTConfig()


        assert (
            self.config.n_embd % self.config.n_head == 0
        )


        # Create Query, Key, Value projections

        self.qkv = nn.Linear(
            self.config.n_embd,
            3 * self.config.n_embd
        )


        # Output projection

        self.projection = nn.Linear(
            self.config.n_embd,
            self.config.n_embd
        )


        self.dropout = nn.Dropout(
            self.config.dropout
        )


        # Prevent looking into future tokens

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    self.config.block_size,
                    self.config.block_size
                )
            )
            .view(
                1,
                1,
                self.config.block_size,
                self.config.block_size
            )
        )



    def forward(self, x):


        batch, tokens, channels = x.shape


        head_dim = (
            channels //
            self.config.n_head
        )


        # Generate Q K V

        qkv = self.qkv(x)


        q, k, v = qkv.split(
            channels,
            dim=2
        )


        # Split into attention heads

        q = q.view(
            batch,
            tokens,
            self.config.n_head,
            head_dim
        ).transpose(1,2)


        k = k.view(
            batch,
            tokens,
            self.config.n_head,
            head_dim
        ).transpose(1,2)


        v = v.view(
            batch,
            tokens,
            self.config.n_head,
            head_dim
        ).transpose(1,2)



        # Attention score

        attention = (
            q @ k.transpose(-2,-1)
        ) / (
            head_dim ** 0.5
        )



        # Causal mask

        attention = attention.masked_fill(
            self.mask[:,:,:tokens,:tokens] == 0,
            float("-inf")
        )



        attention = torch.softmax(
            attention,
            dim=-1
        )


        attention = self.dropout(
            attention
        )


        output = (
            attention @ v
        )


        # Merge heads

        output = output.transpose(
            1,
            2
        ).contiguous().view(
            batch,
            tokens,
            channels
        )


        return self.projection(
            output
        )