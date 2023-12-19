import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords

def replace_pattern(text,pattern,replace = ''):
    # Replace diacritics from the text
    cleaned_text = pattern.sub(replace, text)
    return cleaned_text

def clean(text):
    # remove any brackets that have only numbers inside and remove all numbers 
    reg = r'\(\s*(\d+)\s*\/\s*(\d+)\s*\)|\d+'
    text = replace_pattern(text, re.compile(reg))
    # replace all different types of brackets with a single type
    reg_opening_brackets = r'[\[\{]'
    reg_closing_brackets = r'[\]\}]'
    text = replace_pattern(text, re.compile(reg_opening_brackets), '(')
    text = replace_pattern(text, re.compile(reg_closing_brackets), ')')
    # remove some unwanted characters
    reg = r'[/!\-؛،؟:\.]'
    text = replace_pattern(text, re.compile(reg))
    # remove extra spaces
    reg = r'\s+'
    text = replace_pattern(text, re.compile(reg), ' ')
    return text

def split_words_between_brackets(text):
    # Define a regular expression pattern to match words between brackets
    pattern_in_brackets = re.compile(r'\((.*?)\)')
    # Find all matches in the text
    matches_in_brackets = pattern_in_brackets.findall(text)
    # Join all matches into a single string to form a sentence
    matches_in_brackets = [match.strip() for match in matches_in_brackets]

    # Define a regular expression pattern to match sentences outside brackets
    pattern_outside_brackets = re.compile(r'[^()]+(?=\()|(?<=\))[^()]+')
    # Find all matches in the text
    matches_outside_brackets = pattern_outside_brackets.findall(text)
    matches_outside_brackets = [match.strip() for match in matches_outside_brackets]
    matches_in_brackets.extend(matches_outside_brackets)
    return matches_in_brackets

def remove_diactrics(text):
    # remove diacritics
    reg = r'[\u064B-\u065F\u0670\uFE70-\uFE7F]'
    return replace_pattern(text, re.compile(reg))

def preprocess(text, data_type):
    # data_type can be 'train', 'val', or 'test'
    # clean the text from unwanted characters
    text = clean(text)
    # split the text into sentences
    text = split_words_between_brackets(text)
    # if no text was found, return an empty list
    if len(text) == 0:
        return []
    # save the cleaned text with diacritics to a file 
    with open(f'../dataset/cleaned_{data_type}_data_with_diacritics.txt', 'a+',encoding='utf-8') as f:
        f.write('\n'.join(text))
        f.write('\n')
    # remove diacritics
    text = [remove_diactrics(sentence) for sentence in text]
    # save the cleaned text without diacritics to a file
    with open(f'../dataset/cleaned_{data_type}_data_without_diacritics.txt', 'a+',encoding='utf-8') as f:
        f.write('\n'.join(text))
        f.write('\n')
    return text

def preprocess_data(data_type, limit = 0):
    # data_type can be 'train', 'val', or 'test'
    # delete the output files if exist
    with open(f'../dataset/cleaned_{data_type}_data_with_diacritics.txt', 'w',encoding='utf-8') as f:
        pass
    with open(f'../dataset/cleaned_{data_type}_data_without_diacritics.txt', 'w',encoding='utf-8') as f:
        pass
    sentences = []
    # read the data and clean it and save it to the files
    with open(f'../dataset/{data_type}.txt', 'r',encoding='utf-8') as f:
        lines = [next(f).strip() for _ in range(limit)]
        for line in lines:
            sentences.extend(preprocess(line, data_type))
        
    return sentences

def tokenize(text):
    # tokenize the text
    tokenizer = TreebankWordTokenizer()
    # tokens that have list of sentences and each sentence is a list of words
    sentences = [tokenizer.tokenize(sentence) for sentence in text]
    filtered_sentences = []
    for sentence in sentences:
        filtered_tokens = [token for token in sentence if token not in stopwords.words('arabic')]
        if filtered_tokens != []: filtered_sentences.append(filtered_tokens)
    return filtered_sentences