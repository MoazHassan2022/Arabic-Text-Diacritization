import re

def replace_pattern(text,pattern,replace = ''):
    """
    This function replaces a pattern in a text with a string
    Args:
        text: string
        pattern: regex pattern
        replace: string to replace the pattern with
    Returns:
        string
    """
    # Replace diacritics from the text
    cleaned_text = pattern.sub(replace, text)
    return cleaned_text

def clean(lines):
    """
    This function cleans the text from unwanted characters
    Args:
        lines: list of strings
    Returns:
        list of strings
    """
    for i in range(len(lines)):
        # remove any brackets that have only numbers inside and remove all numbers
        reg = r'\(\s*(\d+)\s*\)|\(\s*(\d+)\s*\/\s*(\d+)\s*\)|\d+'
        lines[i] = replace_pattern(lines[i], re.compile(reg))
        # replace all different types of brackets with a single type
        reg_brackets = r'[\[\{\(\]\}\)]'
        lines[i] = re.compile(reg_brackets).sub('', lines[i])
        # remove some unwanted characters
        #reg = r'[/!\-؛،؟:\.]'
        reg = r'[/\/\\\-]'
        lines[i] = replace_pattern(lines[i], re.compile(reg))
        # remove unwanted characters
        reg = r'[,»–\';«*\u200f"\\~`]'
        lines[i] = replace_pattern(lines[i], re.compile(reg))
        # remove extra spaces
        reg = r'\s+'

        lines[i] = replace_pattern(lines[i], re.compile(reg), ' ')
    return lines

def remove_diactrics(lines):
    """
    This function removes diacritics from the text
    Args:
        lines: list of strings
    Returns:
        list of strings
    """
    for i in range(len(lines)):
        # remove diacritics
        reg = r'[\u064B-\u065F\u0670\uFE70-\uFE7F]'
        lines[i] = replace_pattern(lines[i], re.compile(reg))
    return lines

def preprocess(lines, data_type, dataset_path = '../dataset', with_labels = True):
    """
    This function cleans the text and saves it to a file
    Args:
        lines: list of strings
        data_type: 'train', 'val', or 'test'
        dataset_path: path to the dataset files
        with_labels: if True, the labels will be saved to the file
    Returns:
        list of strings
    """
    # data_type can be 'train', 'val', or 'test'
    # clean the text from unwanted characters
    lines = clean(lines)
    if len(lines) == 0:
        return lines
    if with_labels:
        # save the cleaned text with diacritics to a file
        with open(f'{dataset_path}/cleaned_{data_type}_data_with_diacritics.txt', 'a+',encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n')
    # remove diacritics
    lines = remove_diactrics(lines)
    # save the cleaned text without diacritics to a file
    with open(f'{dataset_path}/cleaned_{data_type}_data_without_diacritics.txt', 'a+',encoding='utf-8') as f:
        f.write('\n'.join(lines))
        f.write('\n')
    return lines

def preprocess_data(data_type, limit = None, dataset_path = '../dataset', with_labels = True):
    """
    This function reads the data and cleans it and saves it to files
    Args:
        data_type: 'train', 'val', or 'test'
        limit: number of lines to read
        dataset_path: path to the dataset files
        with_labels: if True, the labels will be saved to the file
    Returns:
        list of strings
    """
    # data_type can be 'train', 'val', or 'test'
    # delete the output files if exist
    with open(f'{dataset_path}/cleaned_{data_type}_data_with_diacritics.txt', 'w',encoding='utf-8') as f:
        pass
    with open(f'{dataset_path}/cleaned_{data_type}_data_without_diacritics.txt', 'w',encoding='utf-8') as f:
        pass
    sentences = []
    # read the data and clean it and save it to the files
    with open(f'{dataset_path}/{data_type}.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        if limit == None:
            limit = len(lines)
        lines = lines[:limit]
        sentences = preprocess(lines, data_type, dataset_path, with_labels=with_labels)

    return sentences
