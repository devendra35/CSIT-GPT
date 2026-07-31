import re
import unicodedata
from typing import Dict

# CSIT DOCUMENT STRUCTURE


IMPORTANT_HEADERS = [
    "Subject",
    "Course",
    "Semester",
    "Chapter",
    "Unit",
    "Topic",
    "Module",
    "Lesson",
    "Section",
    "Introduction",
    "Conclusion",
]


class TextCleaner:
    """
    A configurable, deterministic cleaning pipeline.

    Designed for:
        - Nepali CSIT notes
        - PDF extracted books
        - Question papers
        - Lab manuals

    Usage:
        cleaner = TextCleaner()
        clean_text = cleaner.clean("DBMS    is   a database!!!")
    """


    PRESERVE_CHARS = set("{}[]()<>=+-*/%^&|~!@#$_.,;:'\"?\\")


    # Nepali sentence markers
    NEPALI_PUNCT = set("।॥")


    def __init__(
        self,
        lowercase_english: bool = False,
        remove_duplicate_lines: bool = True,
        normalize_form: str = "NFC",
    ):


        self.lowercase_english = lowercase_english
        self.remove_duplicate_lines = remove_duplicate_lines
        self.normalize_form = normalize_form



        self.important_headers = IMPORTANT_HEADERS

        self.stats = {}



      


        self._multi_space_re = re.compile(
            r"[ \t]{2,}"
        )


        self._multi_newline_re = re.compile(
            r"\n{3,}"
        )


        # PDF artifact:
    

        self._hyphen_linebreak_re = re.compile(
            r"(\w)-\n(\w)"
        )


        self._punct_run_re = re.compile(
            r"([.\-_=*#])\1{3,}"
        )


        self._control_char_re = re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
        )

    # Unicode Processing



    def normalize_unicode(
        self,
        text: str
    ) -> str:

        return unicodedata.normalize(
            self.normalize_form,
            text
        )



    def remove_control_characters(
        self,
        text: str
    ) -> str:

        return self._control_char_re.sub(
            "",
            text
        )


    # PDF Cleaning


    def fix_pdf_linebreaks(
        self,
        text: str
    ) -> str:

        return self._hyphen_linebreak_re.sub(
            r"\1\2",
            text
        )

    # CSIT Structure Preservation (NEW


    def preserve_headers(
        self,
        text: str
    ) -> str:
        """
        Preserve CSIT document hierarchy.

        Example:

        Subject: DBMS

        Unit: Normalization

        Topic: Functional Dependency

        This helps future:
            - RAG search
            - semester filtering
            - AI tutoring
        """


        lines = text.split("\n")

        output = []


        for line in lines:

            stripped = line.strip()


            if any(
                stripped.startswith(header)
                for header in self.important_headers
            ):

                output.append(stripped)

            else:

                output.append(line)


        return "\n".join(output)



    # Formatting


    def collapse_whitespace(
        self,
        text: str
    ) -> str:


        text = self._multi_space_re.sub(
            " ",
            text
        )


        text = self._multi_newline_re.sub(
            "\n\n",
            text
        )


        return "\n".join(
            line.rstrip()
            for line in text.split("\n")
        ).strip()



    def collapse_punct_runs(
        self,
        text: str
    ) -> str:

        return self._punct_run_re.sub(
            r"\1\1\1",
            text
        )

    # Duplicate Removal



    def dedupe_lines(
        self,
        text: str
    ) -> str:


        seen = set()

        out_lines = []


        for line in text.split("\n"):

            stripped = line.strip()


            if stripped == "":

                out_lines.append(line)

                continue


            if stripped in seen:

                continue


            seen.add(stripped)

            out_lines.append(line)


        return "\n".join(out_lines)



    # Symbol Cleaning



    def remove_unwanted_symbols(
        self,
        text: str
    ) -> str:


        allowed = (
            self.PRESERVE_CHARS
            |
            self.NEPALI_PUNCT
        )


        out_chars = []


        for ch in text:


            category = unicodedata.category(ch)


            # Nepali matra + virama protection
            is_mark = category.startswith("M")


            if (
                ch.isalnum()
                or is_mark
                or ch.isspace()
                or ch in allowed
            ):

                out_chars.append(ch)


        return "".join(out_chars)



    # Statistics (NEW)


    def get_statistics(
        self,
        text: str
    ) -> Dict:


        return {

            "characters": len(text),

            "words": len(text.split()),

            "lines": len(text.splitlines())

        }



    # Main Pipeline


    def clean(
        self,
        text: str
    ) -> str:


        before = self.get_statistics(text)



        text = self.normalize_unicode(text)

        text = self.remove_control_characters(text)

        text = self.fix_pdf_linebreaks(text)


        # NEW:
        # Preserve CSIT hierarchy before symbol filtering

        text = self.preserve_headers(text)


        text = self.remove_unwanted_symbols(text)

        text = self.collapse_punct_runs(text)

        text = self.collapse_whitespace(text)



        if self.lowercase_english:

            text = "".join(
                c.lower()
                if c.isascii()
                else c
                for c in text
            )



        if self.remove_duplicate_lines:

            text = self.dedupe_lines(text)



        after = self.get_statistics(text)


        self.stats = {

            "before": before,

            "after": after

        }



        return text

    # File Processing


    def clean_file(
        self,
        in_path: str,
        out_path: str
    ) -> None:


        with open(
            in_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            raw = f.read()



        cleaned = self.clean(raw)



        with open(
            out_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(cleaned)




# Testing


if __name__ == "__main__":


    cleaner = TextCleaner()


    sample = """
Subject: Database Management System

Semester: 4

Unit: Normalization


DBMS    is   important!!!

DBMS    is   important!!!

डाटाबेस व्यवस्थापन प्रणाली

int x = 10;
"""


    result = cleaner.clean(sample)


    print("\nCLEAN OUTPUT")
    print("----------------")

    print(result)


    print("\nSTATISTICS")
    print("----------------")

    print(cleaner.stats)
