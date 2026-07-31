"""
CSIT-GPT Text Generation Engine
"""

import torch
from tokenizers import Tokenizer

from model.gpt import GPT


# CONFIGURATION

MODEL_PATH = "csit_gpt.pth"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

MAX_TOKENS = 50
BLOCK_SIZE = 32


# DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)




# LOAD TOKENIZER


try:

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    print(" Tokenizer loaded")


except Exception as e:

    print(" Tokenizer loading failed:")
    print(e)
    exit()




# LOAD MODEL


try:

    model = GPT()


    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )


    model.to(device)

    model.eval()


    print(" Model loaded")


except Exception as e:

    print(" Model loading failed:")
    print(e)
    exit()



print("\n CSIT-GPT Ready\n")



# TEXT GENERATION



def generate(
        prompt,
        max_tokens=MAX_TOKENS
):


    # Encode prompt

    encoded = tokenizer.encode(
        prompt
    )


    tokens = encoded.ids



    if len(tokens) == 0:

        return "Please enter a valid prompt."



    x = torch.tensor(
        tokens,
        dtype=torch.long
    ).unsqueeze(0).to(device)



    generated = x



    # Generate tokens

    for _ in range(max_tokens):


        # Keep context size limited

        context = generated[:, -BLOCK_SIZE:]



        with torch.no_grad():


            logits = model(
                context
            )



        # Last token prediction

        logits = logits[:, -1, :]



        # Greedy decoding

        next_token = torch.argmax(
            logits,
            dim=-1
        )



        generated = torch.cat(

            [
                generated,
                next_token.unsqueeze(0)
            ],

            dim=1

        )



    # Decode

    output = tokenizer.decode(
        generated[0].tolist()
    )



    # Remove tokenizer artifacts

    output = (
        output
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
    )



    # Remove extra spaces

    output = " ".join(
        output.split()
    )



    return output




# CHAT LOOP



while True:


    prompt = input(
        "You: "
    )



    if prompt.lower() in [
        "exit",
        "quit"
    ]:

        print(
            " CSIT-GPT shutting down"
        )

        break



    response = generate(
        prompt
    )


    print(
        "\nCSIT-GPT:",
        response,
        "\n"
    )