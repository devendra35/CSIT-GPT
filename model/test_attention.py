import torch

from attention import SelfAttention



model = SelfAttention()


x = torch.randn(
    2,
    10,
    256
)


output = model(x)


print(
    "Input:",
    x.shape
)


print(
    "Output:",
    output.shape
)