import spacy
from spacy_langdetect import LanguageDetector
import os

def sort_by_language():
    """Sort .txt files by document average language, implemented using spacy"""
    curr_dir=os.getcwd()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    for filename in os.listdir(curr_dir):
        if filename.endswith(".txt"): 
            f = open(filename, "r",encoding="utf8")
            text=f.read()
            f.close()
            doc = nlp(text)
            lang=doc._.language['language']
            if not os.path.exists(os.path.join(curr_dir,lang)):
                os.makedirs(os.path.join(curr_dir,lang))
            else:
                print("Moving",filename,'to',lang)
                os.replace(os.path.join(curr_dir,filename), os.path.join(curr_dir,lang,filename))
        else:
            continue