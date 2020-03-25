import spacy
from spacy_langdetect import LanguageDetector
import os

def language_detector():
    """
    Sort .txt files by document average language, by moving them into respective subfolders, implemented using SpaCy.
    Subfolders with the language as the name of the folders are created for every unseen language.
    Supports multiple languages. 
    """
    curr_dir = os.getcwd()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    for filename in os.listdir(curr_dir):
        if filename.endswith(".txt"): 
            f = open(filename, "r",encoding="utf8")
            text=f.read()
            f.close()
            doc = nlp(text)
            lang=doc._.language['language']
            if not os.path.exists(os.path.join(curr_dir, lang)):
                os.makedirs(os.path.join(curr_dir,lang))
            else:
                print(f"Moving {filename} to {lang}")
                os.replace(os.path.join(curr_dir, filename), os.path.join(curr_dir, lang, filename))
        else:
            continue

if __name__ == "__main__":
    language_detector()