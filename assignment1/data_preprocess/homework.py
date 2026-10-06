import argparse
import re
import requests
import json
from utils import  read_warc_file, read_wet_file
from datasets import load_dataset
from typing import Set, Dict
import string
from bs4 import BeautifulSoup

# we have todos below
def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open('./bad_word_list.txt', 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)


def html_to_text(html) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """

    # ignore broken bytes instead of throwing error?
    if isinstance(html, bytes):
        html = html.decode('utf-8', errors='ignore')

    # use beqautifulsoup? check guidebook
    soup = BeautifulSoup(html, 'html.parser')

    # delete tags whose contents are not article text?
    # do we need more?
    for tag in soup(['script', 'style', 'head', 'noscript']):
        tag.decompose()
    
    #extract visible text, new line between block?
    text = soup.get_text(separator='\n')
    return text.strip() # remove blank space at start and end

    # pass 

def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated. ?
    """
    # Replace US social security numbers (XXX-XX-XXXX format)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', text)
    text = re.sub(r'\+1\d{10}\b', '+' + 'X' * 11, text)
    return text
    # do we need this as well?
    # pass 
    

def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """

    kept = []
    for paragraph in text.split("\n"):
        if re.search(r'[A-Za-z0-9]{101,}', paragraph):
            continue                                         # junk? -> skip
        if not any(char in string.punctuation for char in paragraph):
            continue                                         # no punctuation -> skip
        kept.append(paragraph)                               
    return "\n".join(kept)         
    # pass


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    # pass 
        # gate 1: bad words? isn't this checked in clean_text? but we can check again
    lowered = text.lower()
    for bad_word in retrieve_bad_words():
        if bad_word in lowered:
            return False

    # gate 2: punctuation
    if not any(char in string.punctuation for char in text):
        return False

    # gate 3: something that is not whitespace
    if len(text.strip()) == 0:
        return False

    # gate 4: >=80% of characters are alphanumeric/punctuation/whitespace
    allowed = string.ascii_letters + string.digits + string.punctuation + string.whitespace
    allowed_count = sum(1 for char in text if char in allowed)
    if allowed_count / len(text) < 0.8:
        return False

    return True


def is_english_text(text: str) -> bool:
    """Detects if text is primarily in English based on character distribution.
    Args:
        text (str): Text to analyze
    Returns:
        bool: True if text is primarily English, False otherwise
    """
    # pass
    letters = [c for c in text if c.isalpha()]          # keep only letters
    if len(letters) < 20:                               # too short to judge
        return False

    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    if ascii_letters / len(letters) < 0.8:              # not Latin script
        return False

    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return False

    common = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that',
              'have', 'it', 'for', 'not', 'on', 'with', 'is', 'are', 'was', 'were'}
    found = {w.lower() for w in words} & common          # & = what they share
    return len(found) >= 2
    

def deduplicate_texts(texts: list[str]) -> list[str]:
    """Deduplicates text by removing duplicate sentences.
    Args:
        texts (list[str]): List of text strings to deduplicate.
    Returns:
        list[str]: Deduplicated list of texts. Implemented a simple Jaccard similarity based deduplication.
    """
    # pass
    def words(t):
        return set(re.findall(r"[a-z0-9']+", t.lower()))   # words as a set

    kept_texts = []
    kept_word_sets = []
    for text in texts:
        word_set = words(text)
        duplicate = False
        for seen in kept_word_sets:
            union = word_set | seen                          # | = all words together
            if len(union) == 0 or len(word_set & seen) / len(union) >= 0.5:
                duplicate = True
                break                                        # stop checking
        if not duplicate:
            kept_texts.append(text)
            kept_word_sets.append(word_set)
    return kept_texts


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str,  default = '', help = 'Specify the path for your warc file.')
    parser.add_argument('--dfname', type = str,  default = '', help = 'Specify the path where you stored topic_dataset.json')
    parser.add_argument('--num_records', type = int,  default=30, help = 'Specify the number of records you want to parse (only used for debugging with smaller sets)')
    parser.add_argument('--output', type = str,  default='cleaned_documents.txt', help = 'Output file for cleaned text documents')
    # parser.add_argument('--wet_name', type = str, default = '', help = 'Specify the path for your wet file.')
    args = parser.parse_args()

    if args.fname:
        seen = 0
        passes = 0

        with open(args.output, 'w', encoding='utf-8') as output_file:
            for url, html_text in read_warc_file(args.fname, args.num_records):
                seen += 1
                # print("Before HTML to text: ", str(html_text))
                text = html_to_text(html_text)
                # print("\n\n\nAfter HTML to text: ", text)
                cleaned_text = clean_text(text)
                # print("After cleaning: ", cleaned_text)
                cleaned_nopii_text = replace_pii(cleaned_text)
                # print("After PII removal: ", cleaned_nopii_text)
                passes_check = heuristic_quality_filter(cleaned_nopii_text)
                is_english = is_english_text(cleaned_nopii_text)
                print(url)
                print("Passes heuristic quality filter:", passes_check)
                print("Is English text:", is_english)
                if passes_check and is_english:
                    passes += 1
                    # Replace newlines with spaces to keep each document on one line
                    single_line_text = cleaned_nopii_text.replace('\n', ' ').replace('\r', ' ').strip()
                    output_file.write(single_line_text + '\n')
                    print("Saved cleaned English document to output file")
                elif passes_check and not is_english:
                    print("Document filtered out: not English")

        print(f"{passes} passed out of {seen} records processed.")
        print(f"Cleaned documents saved to: {args.output}")

    if args.dfname:
        with open(args.dfname, 'r') as f:
            raw_texts = json.load(f)
        raw_texts = [item['text'] for item in raw_texts['data']]
        deduplicated_texts = deduplicate_texts(raw_texts)
        print(f"{len(deduplicated_texts)} deduplicated out of {len(raw_texts)} records processed.")
    else:
        print("Usage: python homework.py --fname data.warc")