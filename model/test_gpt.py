import torch

from gpt import GPT


model = GPT()


tokens = torch.randint(
    0,
    8000,
    (2,10)
)


output = model(tokens)


print("Input:")
print(tokens.shape)


print("Output:")
print(output.shape)