"""

CSIT-GPT Dataset Pipeline

Raw Notes
    ↓
Load Text
    ↓
Clean Text
    ↓
Save Training Dataset
"""


from pathlib import Path
from cleaner import TextCleaner



class CSITDatasetLoader:


    def __init__(
        self,
        raw_dir="data/raw",
        cleaned_dir="data/cleaned"
    ):

        self.raw_dir = Path(raw_dir)

        self.cleaned_dir = Path(cleaned_dir)

        self.cleaned_dir.mkdir(
            exist_ok=True
        )


        self.cleaner = TextCleaner()



    def load_documents(self):

        documents = []


        files = list(
            self.raw_dir.glob("*.txt")
        )


        if not files:

            print(
                "No .txt files found in data/raw"
            )

            return documents



        for file in files:

            print(
                f"Loading: {file.name}"
            )


            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )


            documents.append(text)


        return documents



    def process(self):


        documents = self.load_documents()


        cleaned_documents = []


        for document in documents:


            cleaned = self.cleaner.clean(
                document
            )


            cleaned_documents.append(
                cleaned
            )



        return cleaned_documents



    def save(self):


        cleaned = self.process()


        if not cleaned:

            print(
                "Nothing to save"
            )

            return



        output = "\n\n".join(
            cleaned
        )


        output_path = (
            self.cleaned_dir /
            "training.txt"
        )


        output_path.write_text(
            output,
            encoding="utf-8"
        )


        print("\nDataset Ready ")

        print(
            f"Saved: {output_path}"
        )



if __name__ == "__main__":


    loader = CSITDatasetLoader()

    loader.save()
