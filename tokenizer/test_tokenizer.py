from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)


texts = [
    "Database Management System",
    "डाटाबेस व्यवस्थापन प्रणाली",
    "SELECT * FROM student;"
]


for text in texts:

    output = tokenizer.encode(text)

    print("\nTEXT:")
    print(text)

    print("TOKENS:")
    print(output.tokens)

    print("IDS:")
    print(output.ids)