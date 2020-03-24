import wikipedia
import sys
import os

from subprograms.config import params


def scraper():
    """
    Extract wikipedia page summary and convert it to .txt files.
    - config.py contains search terms, number of results & languages that can be altered.
    """
    languages, search_term, num_results = params['languages'], params['search_term'], params['num_results']

    for lang in languages:
        for term in search_term:
            wikipedia.set_lang(lang)
            page_list=wikipedia.search(term, results=num_results)
            print(f"Found {len(page_list)} Articles")
            for page in page_list:
                try:
                    words=wikipedia.summary(page)
                    if words:
                        print(f"Saving {str(page)}.txt")
                        file_dir = os.path.join(os.getcwd(), 'files',str(page)+'.txt')
                        with open(file_dir, 'w',encoding='utf-8') as outfile:
                            outfile.write(words)
                except:
                    print(f"Error, {page} not downloaded.")
                    break


if __name__ == "__main__":
    scraper()
