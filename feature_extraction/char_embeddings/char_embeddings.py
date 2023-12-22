# Import necessary libraries
import torch
import torch.nn as nn
import pickle
import re
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from preprocessing.preprocess import preprocess_data

def create_char_embeddings(limit = 1000, dataset_path='../../dataset', embedding_dim = 100):
    # preprocess and save the data
    preprocess_data(data_type='train', limit=limit, dataset_path=dataset_path)
    preprocess_data(data_type='val', limit=limit, dataset_path=dataset_path)
    
    # load data
    with open(f'{dataset_path}/cleaned_train_data_without_diacritics.txt', 'r', encoding='utf-8') as file:
        # read all lines into a single string
        training_data = re.compile(r'[\n\r\t\s]').sub('', file.read())
    with open(f'{dataset_path}/cleaned_val_data_without_diacritics.txt', 'r', encoding='utf-8') as file:
        # read all lines into a single string
        validation_data = re.compile(r'[\n\r\t\s]').sub('', file.read())

    # Tokenize the text into sequences at the character level
    vocab = set(''.join(training_data + validation_data))

    char_to_index = {char: idx + 1 for idx, char in enumerate(vocab)}
    index_to_char = {idx + 1: char for idx, char in enumerate(vocab)}
    
    # Create the embedding layer
    embedding = nn.Embedding(len(vocab), embedding_dim)
    # Get sequences of unique chars
    sequences = torch.tensor([idx for idx, _ in index_to_char.items()])
    # Apply the embedding layer to get the embedding vectors
    embedding_vectors = embedding(sequences)
    
    # save the embedding vectors
    with open(f'{dataset_path}/char_embedding_vectors.pkl', 'wb') as file:
        pickle.dump(embedding_vectors, file)
        
    return embedding_vectors, char_to_index, index_to_char
    
