import torch

from transformer import TransformerBlock



model = TransformerBlock()


x = torch.randn(
    2,
    10,
    256
)


out = model(x)


print("Input:", x.shape)
print("Output:", out.shape)
