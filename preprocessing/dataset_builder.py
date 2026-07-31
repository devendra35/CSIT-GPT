"""
CSIT-GPT Dataset Builder

Combines raw CSIT educational files,
cleans them,
adds metadata,
and creates training corpus.
"""


import os
from pathlib import Path

from cleaner import TextCleaner



# CONFIG


RAW_DIR = "../data/raw"

OUTPUT_FILE = "../data/cleaned/training.txt"


SUPPORTED_EXTENSIONS = [
    ".txt",
    ".md"
]




# SUBJECT DETECTION



def detect_subject(filename):

    name = filename.lower()


    subjects = {

        "dbms": "Database Management System",
        "os": "Operating System",
        "cn": "Computer Networks",
        "dsa": "Data Structure and Algorithm",
        "daa": "Design and Analysis of Algorithm",
        "ai": "Artificial Intelligence",
        "web": "Web Technology",
        "crypto": "Cryptography",
        "toc": "Theory of Computation",
        "statistics": "Statistics"

    }


    for key,value in subjects.items():

        if key in name:
            return value


    return "General CSIT"



# ==
# BUILDER
# ==


def build_dataset():


    cleaner = TextCleaner()


    raw_path = Path(RAW_DIR)



    if not raw_path.exists():

        print(" Raw data folder missing")

        return



    files = list(
        raw_path.rglob("*")
    )


    text_files = [

        f for f in files

        if f.suffix.lower()
        in SUPPORTED_EXTENSIONS

    ]



    if not text_files:

        print(
            "❌ No training files found"
        )

        return



    dataset = []



    print("\n Building CSIT Dataset\n")



    for file in text_files:


        print(
            "Processing:",
            file.name
        )


        try:


            with open(
                file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:


                content = f.read()



            cleaned = cleaner.clean(
                content
            )



            if len(cleaned.strip()) < 50:

                print(
                    "Skipped (too small)"
                )

                continue



            subject = detect_subject(
                file.name
            )



            formatted = f"""
<subject>
{subject}

<content>
{cleaned}

"""



            dataset.append(
                formatted
            )



        except Exception as e:


            print(
                "Error:",
                file.name,
                e
            )



    os.makedirs(
        "../data/cleaned",
        exist_ok=True
    )



    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:


        f.write(
            "\n".join(dataset)
        )



    print("\n DATASET READY")
    print(
        "Saved:",
        OUTPUT_FILE
    )

    print(
        "Documents:",
        len(dataset)
    )

# RUN


if __name__ == "__main__":

    build_dataset()
