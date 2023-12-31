# Import necessary libraries
import torch
import torch.nn as nn
import pickle
import numpy as np
import re
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from torch.nn import ReLU
from tqdm import tqdm
from torch.utils.data import TensorDataset
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(device)

class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(RNNModel, self).__init__()
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                output, _ = self.rnn(x)
                output = self.fc(output[:, -1, :])
                return output

class RNN():
    def __init__(self):
        # define the diacritics unicode and their corresponding labels classes indices
        self.labels = {
            # no diacritic
            0: 0,
            # fath
            1614: 1,
            # damm
            1615: 2,
            # kasr
            1616: 3,
            # shadd
            1617: 4,
            # sukun
            1618: 5,
            # tanween bel fath
            1611: 6,
            # tanween bel damm
            1612: 7,
            # tanween bel kasr
            1613: 8,
            # shadd and fath
            (1617, 1614): 9,
            # shadd and damm
            (1617, 1615): 10,
            # shadd and kasr
            (1617, 1616): 11,
            # shadd and tanween bel fath
            (1617, 1611): 12,
            # shadd and tanween bel damm
            (1617, 1612): 13,
            # shadd and tanween bel kasr
            (1617, 1613): 14
        }

        self.indicies_to_labels = {
            # no diacritic
            0: 0,
            # fath
            1: 1614,
            # damm
            2: 1615,
            # kasr
            3: 1616,
            # shadd
            4: 1617,
            # sukun
            5: 1618,
            # tanween bel fath
            6: 1611,
            # tanween bel damm
            7: 1612,
            # tanween bel kasr
            8: 1613,
            # shadd and fath
            9: (1617, 1614),
            # shadd and damm
            10: (1617, 1615),
            # shadd and kasr
            11: (1617, 1616),
            # shadd and tanween bel fath
            12: (1617, 1611),
            # shadd and tanween bel damm
            13: (1617, 1612),
            # shadd and tanween bel kasr
            14: (1617, 1613)
        }
        # load the data
        self.load_data()
        
    def load_data(self):
        # load data
        with open('dataset/cleaned_train_data_with_diacritics.txt', 'r', encoding='utf-8') as file:
            self.training_data_diacritized = re.compile(r'[\n\r\t]').sub('', file.read())
        with open('dataset/cleaned_train_data_without_diacritics.txt', 'r', encoding='utf-8') as file:
            self.training_data = re.compile(r'[\n\r\t]').sub('', file.read())
        with open('dataset/cleaned_val_data_with_diacritics.txt', 'r', encoding='utf-8') as file:
            self.validation_data_diacritized = re.compile(r'[\n\r\t]').sub('', file.read())
        with open('dataset/cleaned_val_data_without_diacritics.txt', 'r', encoding='utf-8') as file:
            self.validation_data = re.compile(r'[\n\r\t]').sub('', file.read())
        # Tokenize the text into sequences at the character level
        self.unique_chars = set(''.join(self.training_data + self.validation_data))
        self.diacritization = list(self.labels.keys())
        self.char_to_index = {char: idx for idx, char in enumerate(self.unique_chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.unique_chars)}
        self.train_sequence = self.text_to_sequence(self.training_data)
        self.validation_sequences = self.text_to_sequence(self.validation_data)
    
    def text_to_sequence(self,text):
        return [self.char_to_index[char] for char in text]
    
    
    def extract_labels(self,data_type:str = 'training'):
        labels = []
        data = self.training_data_diacritized if data_type == 'training' else self.validation_data_diacritized
        training_size = len(data)
        index = 0
        while index < training_size:
            if ord(data[index]) not in self.labels:
                # char is not a diacritic
                if (index + 1) < training_size and ord(data[index + 1]) in self.labels:
                    # char has a diacritic
                    if ord(data[index + 1]) == 1617:
                        # char has a shadd diacritic
                        if (index + 2) < training_size and ord(data[index + 2]) in self.labels:
                            # char has a shadd and another diacritic
                            labels.append(self.labels[(1617, ord(data[index + 2]))])
                            # skip next 2 diacritics chars
                            index += 3  # increment by 3 to skip two diacritic chars
                            continue
                        else:
                            # char has a shadd and no other diacritic
                            labels.append(self.labels[1617])
                            # skip next diacritic char
                            index += 2
                            continue
                    # char has a diacritic other than shadd
                    labels.append(self.labels[ord(data[index + 1])])
                    # skip next diacritic char
                    index += 2  # increment by 2 to skip one diacritic char
                    continue
                else:
                    # char has no diacritic
                    labels.append(0)
            index += 1  # increment by 1 for normal iteration
        return labels
    
    
    def train(self,vectorizer):
        self.vectorizer = vectorizer
        char_sequence = [char for char in self.training_data]
        training_data_labels = self.extract_labels('training')
        X_ngrams_training = self.vectorizer.fit_transform(char_sequence)
        labels_training = torch.tensor(training_data_labels)
        X_ngrams_training = torch.tensor(X_ngrams_training.toarray(), dtype=torch.float32)
        # Hyperparameters
        input_size_seq = 72  # Each character is represented as a single input
        hidden_size_seq = 64
        output_size = 15
        num_epochs = 1
        train_dataset = TensorDataset(X_ngrams_training, labels_training)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = RNNModel(input_size_seq,hidden_size_seq,output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):
            for train_sequence, train_labels in train_loader:
                optimizer.zero_grad()
                # Expand dimensions for sequence input (batch size, sequence length, input size)
                batch_ngrams_expanded = train_sequence.unsqueeze(1)
                # Forward pass
                outputs = model(batch_ngrams_expanded)
                # Compute loss
                loss = criterion(outputs, train_labels)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        
        # save the model 
        torch.save(model, 'RNN_TF-IDF.pth')
    
    def validate(self):
        # load the model
        loaded = torch.load('model2.pth')
        char_sequence_validation = [char for char in self.validation_data]
        validation_data_labels = self.extract_labels('validation')
        X_ngrams_validation = self.vectorizer.transform(char_sequence_validation)
        labels_validation = torch.tensor(validation_data_labels)
        X_ngrams_validation = torch.tensor(X_ngrams_validation.toarray(), dtype=torch.float32)
    
        with torch.no_grad():
            loaded.eval()
            test_inputs = torch.tensor(X_ngrams_validation)
            test_labels = torch.tensor(labels_validation)
            test_inputs = test_inputs.unsqueeze(1)
            test_outputs = loaded(test_inputs.float())
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == test_labels).sum().item() / len(test_labels)
        print("Test Accuracy:", accuracy)


if __name__ == "__main__":
    rnn = RNN()
    rnn.load_data()
    vectorizer = None
    if  len(sys.argv) == 1 or sys.argv[1] == 'tfidf':
        print("Training with TF-IDF vectorizer")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,2))
    elif sys.argv[1] == 'bigram':
        print("Training with bigram vectorizer")
        vectorizer = CountVectorizer(ngram_range=(2, 2),analyzer='char_wb')
    else:
        raise Exception("Invalid argument")
    rnn.train(vectorizer=vectorizer)
    rnn.validate()