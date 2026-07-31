"""
CSIT-GPT Transformer Block
"""

import torch
import torch.nn as nn

from model.attention import SelfAttention
from model.config import GPTConfig



class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        config = GPTConfig()

        self.network = nn.Sequential(

            nn.Linear(
                config.n_embd,
                4 * config.n_embd
            ),

            nn.GELU(),

            nn.Linear(
                4 * config.n_embd,
                config.n_embd
            ),

            nn.Dropout(
                config.dropout
            )

        )


    def forward(self,x):

        return self.network(x)




class TransformerBlock(nn.Module):


    def __init__(self):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            GPTConfig.n_embd
        )

        self.ln2 = nn.LayerNorm(
            GPTConfig.n_embd
        )


        self.attention = SelfAttention()


        self.feed_forward = FeedForward()



    def forward(self,x):


        # Attention + residual

        x = x + self.attention(
            self.ln1(x)
        )


        # Feed Forward + residual

        x = x + self.feed_forward(
            self.ln2(x)
        )


        return x