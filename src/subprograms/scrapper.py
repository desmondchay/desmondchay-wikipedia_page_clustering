import wikipedia
import sys

def make_corpus(languages,search_term,num_results):
    """Extract wikipedia page summary and convert it to .txt files. config.py contains search terms, number of results & languages that can be altered"""
    for lang in languages:
        for term in search_term:
            wikipedia.set_lang(lang)
            page_list=wikipedia.search(term, results=num_results)
            print('Found',len(page_list),'Articles')
            for page in page_list:
                try:
                    words=wikipedia.summary(page)
                    if words:
                        print("Saving",str(page)+'.txt')
                        with open(str(page)+'.txt', 'w',encoding='utf-8') as outfile:
                            outfile.write(words)
                except:
                    pass