


from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE

from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import ByteLevel

from tokenizers.processors import TemplateProcessing



# Paths

DATA_FILE = "../data/cleaned/training.txt"

OUTPUT_DIR = Path(".")



# Special tokens

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]



def train():


    print(" Starting CSIT-GPT tokenizer training")


    tokenizer = Tokenizer(
        BPE(
            unk_token="<UNK>"
        )
    )


    

    tokenizer.pre_tokenizer = ByteLevel()



    trainer = BpeTrainer(

        vocab_size=8000,

        min_frequency=2,

        special_tokens=SPECIAL_TOKENS

    )



    tokenizer.train(
        [
            DATA_FILE
        ],
        trainer
    )



    tokenizer.post_processor = TemplateProcessing(

        single="<BOS> $A <EOS>",

        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>"))
        ]

    )



    tokenizer.save(
        "tokenizer.json"
    )


    tokenizer.model.save(
        "."
    )


    print(" Tokenizer trained successfully")

    print("Files created:")
    print(
        "tokenizer.json"
    )
    print(
        "vocab.json"
    )
    print(
        "merges.txt"
    )



if __name__ == "__main__":

    train()
