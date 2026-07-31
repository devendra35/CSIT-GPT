"""
CSIT-GPT Training Dataset

Converts text into token sequences
"""

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer



class CSITDataset(Dataset):


    def __init__(
        self,
        text_file,
        tokenizer_file,
        block_size
    ):


        self.block_size = block_size


        # Load tokenizer

        self.tokenizer = Tokenizer.from_file(
            tokenizer_file
        )


        # Load text

        with open(
            text_file,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()



        # Encode text

        encoded = self.tokenizer.encode(
            text
        )


        self.tokens = encoded.ids



    def __len__(self):

        return len(self.tokens) - self.block_size



    def __getitem__(self,index):


        x = self.tokens[
            index:index+self.block_size
        ]


        y = self.tokens[
            index+1:index+self.block_size+1
        ]


        return (
            torch.tensor(x),
            torch.tensor(y)
        )