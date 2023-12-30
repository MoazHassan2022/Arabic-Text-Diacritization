# Import necessary libraries
import textwrap
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from preprocessing.preprocess import preprocess_data

def count_spaces(input_string):
    """
    Count the number of spaces in a string
    Args:
        input_string: string to count spaces in
    Returns:
        space_count: number of spaces in the string
    """
    
    space_count = len(re.findall(r'\s', input_string))
    return space_count

def tokenize_data(data_type='test', dataset_path='../dataset', max_len=200, with_labels=False):
    """
    Tokenize the data into sentences of max_len, without cutting words
    Args:
        data_type: 'train', 'test', or 'val'
        dataset_path: path to the dataset folder
        max_len: maximum length of a sentence
        with_labels: whether to tokenize data with labels or not
    Returns:
        data: list of sentences without diacritics
        data_with_diacritics: list of sentences with diacritics
    """
    
    data = []
    spaces = []
    # tokenize data without diacritics
    with open(f'{dataset_path}/cleaned_{data_type}_data_without_diacritics.txt', 'r', encoding='utf-8') as file:
        # read all lines into array of lines
        data_lines = file.readlines()
        for i in range(len(data_lines)):
            data_lines[i] = re.compile(r'[\n\r\t]').sub('', data_lines[i])
            data_lines[i] = re.compile(r'\s+').sub(' ', data_lines[i])
            data_lines[i] = data_lines[i].strip()

            # split the line into sentences by dot
            dot_splitted_list = data_lines[i].split('.')

            # remove last string if empty
            if dot_splitted_list[-1] == '':
                dot_splitted_list = dot_splitted_list[:-1]

            for dot_splitted in dot_splitted_list:
                dot_splitted = dot_splitted.strip()
                # Split the line into sentences of max_len, without cutting words
                sentences = textwrap.wrap(dot_splitted, max_len)

                for sentence in sentences:
                    data.append(sentence)
                    spaces.append(count_spaces(sentence))
                    
    data_with_diacritics = []
    spaces_index = 0
    # tokenize data with diacritics
    with open(f'{dataset_path}/cleaned_{data_type}_data_with_diacritics.txt', 'r', encoding='utf-8') as file:
        data_with_diacritics_lines = file.readlines()
        for i in range(len(data_with_diacritics_lines)):
            data_with_diacritics_lines[i] = re.compile(r'[\n\r\t]').sub('', data_with_diacritics_lines[i])
            data_with_diacritics_lines[i] = re.compile(r'\s+').sub(' ', data_with_diacritics_lines[i])
            data_with_diacritics_lines[i] = data_with_diacritics_lines[i].strip()

            # split the line into sentences by dot
            dot_splitted_list = data_with_diacritics_lines[i].split('.')

            # remove last string if empty
            if dot_splitted_list[-1] == '':
                dot_splitted_list = dot_splitted_list[:-1]

            for dot_splitted in dot_splitted_list:
                dot_splitted = dot_splitted.strip()
                remaining = dot_splitted
                remaining_length = len(remaining)
                #  cut the line into sentences using previously calculated spaces
                while(remaining_length > 0):
                    spaces_to_include = spaces[spaces_index]
                    spaces_index += 1
                    words = remaining.split()
                    if len(words) <= spaces_to_include + 1:
                        # if the remaining words are less than the spaces to include, add them to the data
                        data_with_diacritics.append(remaining.strip())
                        remaining_length = 0
                        break
                    else:
                        # if the remaining words are more than the spaces to include, add the first words to the data
                        sentence = ' '.join(words[:spaces_to_include + 1])
                        data_with_diacritics.append(sentence.strip())
                        # and keep the remaining words for the next iteration
                        remaining = ' '.join(words[spaces_to_include + 1:]).strip()
                        remaining_length = len(remaining)
                        
    return data, data_with_diacritics

def convert2idx(data=[], char_to_index={}, max_len=200, device='cpu'):
    """
    Convert the data into sequences of indices
    Args:
        data: list of sentences
        char_to_index: dictionary mapping characters to indices
        max_len: maximum length of a sentence
    Returns:
        data_sequences: list of sequences of indices
    """
    
    # build one array that holds all sequences of data
    data_sequences = [[char_to_index[char] for char in sequence] for sequence in data]
    
    # pad sequences to the maximum length
    data_sequences = [sequence + [0] * (max_len - len(sequence)) for sequence in data_sequences]

    # convert to tensor
    data_sequences = torch.tensor(data_sequences).to(device)
    
    return data_sequences

def label_data(data_with_diacritics=[], labels={}, max_len=200, device='cpu'):
    """
    Label the data with diacritics
    Args:
        data_with_diacritics: list of sentences with diacritics
        labels: dictionary mapping diacritics to indices
        max_len: maximum length of a sentence
    Returns:
        data_labels: list of sequences of labels
    """
    data_labels = []
    size = len(data_with_diacritics)
    for sentence_index in range(size):
        sentence_labels = []
        sentence_size = len(data_with_diacritics[sentence_index])
        index = 0
        while index < sentence_size:
            if ord(data_with_diacritics[sentence_index][index]) not in labels:
                # char is not a diacritic
                if (index + 1) < sentence_size and ord(data_with_diacritics[sentence_index][index + 1]) in labels:
                    # char has a diacritic
                    if ord(data_with_diacritics[sentence_index][index + 1]) == 1617:
                        # char has a shadd diacritic
                        if (index + 2) < sentence_size and ord(data_with_diacritics[sentence_index][index + 2]) in labels:
                            # char has a shadd and another diacritic
                            sentence_labels.append(labels[(1617, ord(data_with_diacritics[sentence_index][index + 2]))])
                            # skip next 2 diacritics chars
                            index += 3  # increment by 3 to skip two diacritic chars
                            continue
                        else:
                            # char has a shadd and no other diacritic
                            sentence_labels.append(labels[1617])
                            # skip next diacritic char
                            index += 2
                            continue
                    # char has a diacritic other than shadd
                    sentence_labels.append(labels[ord(data_with_diacritics[sentence_index][index + 1])])
                    # skip next diacritic char
                    index += 2  # increment by 2 to skip one diacritic char
                    continue
                else:
                    # char has no diacritic
                    sentence_labels.append(14)
            index += 1  # increment by 1 for normal iteration

        data_labels.append(sentence_labels)

    # pad sequences to the maximum length
    data_labels = [sequence + [15] * (max_len - len(sequence)) for sequence in data_labels]

    data_labels = torch.tensor(data_labels).to(device)
    
    return data_labels

def get_dataloader(data_type='train', max_len=200, batch_size=256, dataset_path='../dataset', char_to_index={}, labels={}, with_labels=False):
    """
    Get the data loader for the given data type
    Args:
        data_type: 'train', 'test', or 'val'
        max_len: maximum length of a sentence
        batch_size: batch size
        dataset_path: path to the dataset folder
        char_to_index: dictionary mapping characters to indices
        labels: dictionary mapping diacritics to indices
        with_labels: whether to get data with labels or not
    Returns:
        dataloader: data loader for the given data type
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(device)

    # call the preprocess function to preprocess the data
    preprocess_data(data_type=data_type, dataset_path=dataset_path, with_labels=with_labels)

    # call the tokenize function to tokenize the data, this will return two lists, one with the data and the other with the data with diacritics
    data, data_with_diacritics = tokenize_data(data_type=data_type, dataset_path=dataset_path, max_len=max_len, with_labels=with_labels)

    # call the convert2idx function to convert the data to indices
    data_sequences = convert2idx(data=data, char_to_index=char_to_index, max_len=max_len, device=device)

    # call the label_data function to label the data, this will return list of lists, each list is labels indexes for a sentence
    if with_labels:
        data_labels = label_data(data_with_diacritics=data_with_diacritics, labels=labels, max_len=max_len, device=device)

    print(f'{data_type} data shape: ', len(data))
    print(f'{data_type} data sequences shape: ', data_sequences.shape)

    # convert the data to tensors data loader
    if with_labels:
        dataset = TensorDataset(data_sequences, data_labels)
    else:
        dataset = TensorDataset(data_sequences, data_sequences)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader