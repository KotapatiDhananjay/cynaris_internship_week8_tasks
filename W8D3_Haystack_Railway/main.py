from src.create_pdfs import documents
from src.pipeline import initialize
from src.evaluation import run_evaluation
import src.create_pdfs

from src.pipeline import initialize
from src.evaluation import run_evaluation


if __name__ == "__main__":

    print("\nW8D3 HAYSTACK RETRIEVAL PROJECT")
    print("=" * 50)

    print("\nSTEP 1: INITIALIZING HAYSTACK")
    initialize()

    print("\nSTEP 2: RUNNING EVALUATION")
    run_evaluation()

    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print("Check the results folder.")
    
if __name__ == "__main__":

    print("\nW8D3 HAYSTACK RETRIEVAL PROJECT")
    print("=" * 50)

    from src.create_pdfs import DATA
    import os

    os.makedirs(DATA, exist_ok=True)

    from src.create_pdfs import __file__ as pdf_script

    # Generate the five PDF files
    exec(open(pdf_script, encoding="utf-8").read())

    print("\nSTEP 1: INITIALIZING HAYSTACK")
    initialize()

    print("\nSTEP 2: RUNNING BM25 AND DENSE EVALUATION")
    run_evaluation()

    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print("Check the results folder.")