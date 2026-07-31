"""
CSIT-GPT Training Engine
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model.gpt import GPT
from model.dataset import CSITDataset


from torch.optim import AdamW



# Device

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Using:", device)



# Dataset

dataset = CSITDataset(

    text_file="data/cleaned/training.txt",

    tokenizer_file="tokenizer/tokenizer.json",

    block_size=32

)



loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True

)



print(
    "Training samples:",
    len(dataset)
)



# Model

model = GPT()

model.to(device)



# Optimizer

optimizer = AdamW(

    model.parameters(),

    lr=3e-4

)



loss_function = nn.CrossEntropyLoss()



epochs = 20



# Training loop

for epoch in range(epochs):


    total_loss = 0



    for x,y in loader:


        x = x.to(device)

        y = y.to(device)



        logits = model(x)



        loss = loss_function(

            logits.view(-1, logits.size(-1)),

            y.view(-1)

        )



        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



    print(

        f"Epoch {epoch+1}/{epochs} Loss:",
        total_loss

    )



# Save model

torch.save(

    model.state_dict(),

    "csit_gpt.pth"

)


print(
    " CSIT-GPT training complete"
)