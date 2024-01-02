import pickle
import textwrap
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.optim.lr_scheduler import StepLR
import warnings

# best_model path
model_path = 'best_model.pkl'

# these are the indices of the characters in the vocabulary
char_to_index = {'د': 1, '؟': 2, 'آ': 3, 'إ': 4, 'ؤ': 5, 'ط': 6, 'م': 7, '،': 8, 'ة': 9, 'ت': 10, 'ر': 11, 'ئ': 12, 'ا': 13, 'ض': 14, '!': 15, ' ': 16, 'ك': 17, 'غ': 18, 'س': 19, 'ص': 20, 'أ': 21, 'ل': 22, 'ف': 23, 'ظ': 24, 'ج': 25, '؛': 26, 'ن': 27, 'ع': 28, 'ب': 29, 'ث': 30, 'ه': 31, 'خ': 32, 'ى': 33, 'ء': 34, 'ز': 35, 'ق': 36, 'ي': 37, 'ش': 38, 'ح': 39, ':': 40, 'ذ': 41, 'و': 42, '.': 43}
index_to_char = {1: 'د', 2: '؟', 3: 'آ', 4: 'إ', 5: 'ؤ', 6: 'ط', 7: 'م', 8: '،', 9: 'ة', 10: 'ت', 11: 'ر', 12: 'ئ', 13: 'ا', 14: 'ض', 15: '!', 16: ' ', 17: 'ك', 18: 'غ', 19: 'س', 20: 'ص', 21: 'أ', 22: 'ل', 23: 'ف', 24: 'ظ', 25: 'ج', 26: '؛', 27: 'ن', 28: 'ع', 29: 'ب', 30: 'ث', 31: 'ه', 32: 'خ', 33: 'ى', 34: 'ء', 35: 'ز', 36: 'ق', 37: 'ي', 38: 'ش', 39: 'ح', 40: ':', 41: 'ذ', 42: 'و', 43: '.'}

# define the diacritics unicode and their corresponding labels classes indices
# note that index 14 is reserved for no diacritic
labels = {
    # fath
    1614: 0,
    # tanween bel fath
    1611: 1,
    # damm
    1615: 2,
    # tanween bel damm
    1612: 3,
    # kasr
    1616: 4,
    # tanween bel kasr
    1613: 5,
    # sukun
    1618: 6,
    # shadd
    1617: 7,
    # shadd and fath
    (1617, 1614): 8,
    # shadd and tanween bel fath
    (1617, 1611): 9,
    # shadd and damm
    (1617, 1615): 10,
    # shadd and tanween bel damm
    (1617, 1612): 11,
    # shadd and kasr
    (1617, 1616): 12,
    # shadd and tanween bel kasr
    (1617, 1613): 13,
    # no diacritic
    0: 14,
    # padded
    15: 15
}

indicies_to_labels = {
    # fath
    0: 1614,
    # tanween bel fath
    1: 1611,
    # damm
    2: 1615,
    # tanween bel damm
    3: 1612,
    # kasr
    4: 1616,
    # tanween bel kasr
    5: 1613,
    # sukun
    6: 1618,
    # shadd
    7: 1617,
    # shadd and fath
    8: (1617, 1614),
    # shadd and tanween bel fath
    9: (1617, 1611),
    # shadd and damm
    10: (1617, 1615),
    # shadd and tanween bel damm
    11: (1617, 1612),
    # shadd and kasr
    12: (1617, 1616),
    # shadd and tanween bel kasr
    13: (1617, 1613),
    # no diacritic
    14: 0,
    # padded
    15: 15
}

# max sentence length
max_len = 600

# batch size, number of sentences to be processed at once
training_batch_size = 32
validation_batch_size = 256

# change this to the path of the dataset files
dataset_path = ''

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

def preprocess(lines, data_type, dataset_path = '', with_labels = False):
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
        with open(f'{dataset_path}cleaned_{data_type}_data_with_diacritics.txt', 'a+',encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n')
    # remove diacritics
    lines = remove_diactrics(lines)
    # save the cleaned text without diacritics to a file
    with open(f'{dataset_path}cleaned_{data_type}_data_without_diacritics.txt', 'a+',encoding='utf-8') as f:
        f.write('\n'.join(lines))
        f.write('\n')
    return lines

def preprocess_data(data_type, limit = None, dataset_path = '.', with_labels = True):
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
    # delete the output files if exist
    with open(f'{dataset_path}cleaned_{data_type}_data_with_diacritics.txt', 'w',encoding='utf-8') as f:
        pass
    with open(f'{dataset_path}cleaned_{data_type}_data_without_diacritics.txt', 'w',encoding='utf-8') as f:
        pass
    sentences = []
    # read the data and clean it and save it to the files
    with open(f'{dataset_path}{data_type}.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        if limit == None:
            limit = len(lines)
        lines = lines[:limit]
        sentences = preprocess(lines, data_type, dataset_path, with_labels=with_labels)

    return sentences

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

def tokenize_data(data_type='test', dataset_path='.', max_len=200, with_labels=True):
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
    with open(f'{dataset_path}cleaned_{data_type}_data_without_diacritics.txt', 'r', encoding='utf-8') as file:
        # read all lines into array of lines
        data_lines = file.readlines()
        for i in range(len(data_lines)):
            data_lines[i] = re.compile(r'[\n\r\t]').sub('', data_lines[i])
            data_lines[i] = re.compile(r'\s+').sub(' ', data_lines[i])
            data_lines[i] = data_lines[i].strip()

            # Split the line into sentences of max_len, without cutting words
            sentences = textwrap.wrap(data_lines[i], max_len)

            for sentence in sentences:
                data.append(sentence)
                spaces.append(count_spaces(sentence))
                    
    data_with_diacritics = []
    
    if with_labels:
        spaces_index = 0
        # tokenize data with diacritics
        with open(f'{dataset_path}cleaned_{data_type}_data_with_diacritics.txt', 'r', encoding='utf-8') as file:
            data_with_diacritics_lines = file.readlines()
            for i in range(len(data_with_diacritics_lines)):
                data_with_diacritics_lines[i] = re.compile(r'[\n\r\t]').sub('', data_with_diacritics_lines[i])
                data_with_diacritics_lines[i] = re.compile(r'\s+').sub(' ', data_with_diacritics_lines[i])
                data_with_diacritics_lines[i] = data_with_diacritics_lines[i].strip()

                remaining = data_with_diacritics_lines[i]
                remaining_length = len(remaining)
                while(remaining_length > 0):
                    spaces_to_include = spaces[spaces_index]
                    spaces_index += 1
                    words = remaining.split()
                    if len(words) <= spaces_to_include + 1:
                        data_with_diacritics.append(remaining.strip())
                        remaining_length = 0
                        break
                    else:
                        sentence = ' '.join(words[:spaces_to_include + 1])
                        data_with_diacritics.append(sentence.strip())
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
                char_sequence = char_to_index[data_with_diacritics[sentence_index][index]]
                if char_sequence == 2 or char_sequence == 8 or char_sequence == 15 or char_sequence == 16 or char_sequence == 26 or char_sequence == 40 or char_sequence == 43:
                    # unwanted char
                    sentence_labels.append(14)
                    index += 1
                    continue
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

def get_dataloader(data_type='train', max_len=200, batch_size=256, dataset_path='', char_to_index={}, labels={}, device='cpu', with_labels=True):
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
        device: 'cpu' or 'cuda'
    Returns:
        dataloader: data loader for the given data type
    """
    # call the preprocess function to preprocess the data
    preprocess_data(data_type=data_type, dataset_path=dataset_path, with_labels=with_labels)

    # call the tokenize function to tokenize the data, this will return two lists, one with the data and the other with the data with diacritics
    data, data_with_diacritics = tokenize_data(data_type=data_type, dataset_path=dataset_path, max_len=max_len, with_labels=with_labels)

    # call the convert2idx function to convert the data to indices
    data_sequences = convert2idx(data=data, char_to_index=char_to_index, max_len=max_len, device=device)

    # call the label_data function to label the data, this will return list of lists, each list is labels indexes for a sentence
    # in case with_labels = False, the labels will be 0 (dummy) for all characters
    if with_labels:
        data_labels = label_data(data_with_diacritics=data_with_diacritics, labels=labels, max_len=max_len, device=device)
    else:
        data_labels = torch.tensor([[15] * max_len] * len(data_sequences)).to(device)
        
    print(f'{data_type} data shape: ', len(data))
    print(f'{data_type} data sequences shape: ', data_sequences.shape)

    # convert the data to tensors data loader
    dataset = TensorDataset(data_sequences, data_labels)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader

class CharLSTM(nn.Module):
    """
    This class implements the character level BiLSTM model
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, dropout_rate, num_layers=1):
        super(CharLSTM, self).__init__()
        # chars embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # LSTM layers
        # batch_first: it means that the input tensor has its first dimension representing the batch size
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # batch normalization layer, to normalize the hidden states, it simply does the following:
        # x = (x - mean) / std
        # where mean and std are calculated for each hidden state
        self.batchnorm = BatchNorm1d(max_len)

        # output layer, final_output = W * concatenated_hidden_states + bias
        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: input tensor
        Returns:
            output: output tensor
        """
        embedded = self.embedding(x) # batch_size * seq_length * embedding_size
        lstm_out, _ = self.lstm(embedded) # batch_size * seq_length * hidden_size
        lstm_out = self.batchnorm(lstm_out) # batch_size * seq_length * hidden_size
        output = self.output(lstm_out)  # batch_size * seq_length * output_size
        return output
    
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(device)
    
    training_dataloader = get_dataloader(data_type='train', max_len=max_len, batch_size=training_batch_size, dataset_path=dataset_path, char_to_index=char_to_index, labels=labels, device=device, with_labels=True)
    validation_dataloader = get_dataloader(data_type='val', max_len=max_len, batch_size=validation_batch_size, dataset_path=dataset_path, char_to_index=char_to_index, labels=labels, device=device, with_labels=True)
    
    # define the model
    num_layers = 5
    vocab_size = len(char_to_index) + 1 # +1 for the 0 padding
    embedding_size = 300
    output_size = len(labels)
    hidden_size = 256
    lr=0.001
    num_epochs = 19
    dropout_rate = 0.2
    lr_step_size = 5
    lr_gamma = 0.1

    model = CharLSTM(vocab_size, embedding_size, hidden_size, output_size, dropout_rate, num_layers).to(device)
    
    # define the loss function and the optimizer
    # ignore the padding index
    # CrossEntropyLoss simply does the softmax and then the negative log likelihood loss
    criterion = nn.CrossEntropyLoss(ignore_index=15)
    
    # Adam optimizer simply does the gradient descent
    # betas: these are the coefficients used for computing running averages of the gradient and its square in the Adam optimizer. 
    # the first value (0.9) is the exponential decay rate for the running average of gradients, 
    # and the second value (0.999) is the exponential decay rate for the running average of squared gradients.
    # eps: is a small constant added to the denominator to prevent division by zero in the computation of the adaptive learning rates.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    
    # scheduler to decrease the learning rate every lr_step_size epochs
    # gamma: multiplicative factor of learning rate decay
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    for epoch in range(num_epochs):
        # train the model for one epoch
        for batch_sequences, batch_labels in training_dataloader:
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(batch_sequences) # batch_size * seq_length * output_size
            # calculate loss
            # outputs: batch_size, seq_length, output_size
            # labels: batch_size, seq_length
            # reshape (flatten) the outputs and labels to be 2D
            # outputs: batch_size * seq_length, output_size
            # labels: batch_size * seq_length
            flat_outputs = outputs.view(-1, outputs.shape[-1])
            flat_labels = batch_labels.view(-1)
            mask = (flat_labels != 15)
            loss = criterion(flat_outputs[mask], flat_labels[mask])
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()

        last_loss = loss.item()

        # validate the model
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        for validation_batch_sequences, validation_batch_labels in validation_dataloader:
            # make the model predict
            outputs = model(validation_batch_sequences) # batch_size * seq_length * output_size
            # calculate accuracy
            predicted_labels = outputs.argmax(dim=2)  # Get the index with the maximum probability
            # ignore the padding index, and the chars like ' ', '؟', ...
            mask = (validation_batch_labels != 15) & (validation_batch_sequences != 2) & (validation_batch_sequences != 8) & (validation_batch_sequences != 15) & (validation_batch_sequences != 16) & (validation_batch_sequences != 26) & (validation_batch_sequences != 40) & (validation_batch_sequences != 43)
            #mask = (validation_batch_labels != 15) & (validation_batch_sequences != 14)
            # sum the correct predictions
            correct_predictions += ((predicted_labels == validation_batch_labels) & mask).sum().item()
            # sum the total predictions (without the padding index, and the chars like ' ', '؟', ...)
            total_predictions += mask.sum().item()

        # return the model to train mode
        model.train()
        
        # decrease the learning rate
        scheduler.step()
        
        # calculate accuracy of the epoch
        accuracy = correct_predictions / total_predictions

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {last_loss * 1:.7f}, Accuracy: {accuracy * 100:.3f}%')
        
    # save the model in pkl file
    file_path = f"BiLSTM_Loss={last_loss * 1:.7f}_Accuracy={accuracy * 100:.3f}%_embedding_size={embedding_size}hidden_size={hidden_size}lr={lr}num_layers={num_layers}num_epochs={num_epochs}max_len={max_len}batch_size={training_batch_size}.pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(model_path):
    """
    Load the model from the given path
    Args:
        model_path: path to the model file
    Returns:
        model: the loaded model
    """
    # load the model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def predict_test(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(device)
    
    test_dataloader = get_dataloader(data_type='test', max_len=max_len, batch_size=validation_batch_size, dataset_path=dataset_path, char_to_index=char_to_index, labels=labels, device=device, with_labels=False)
        
    # open csv file to write the predictions to, with the first row as the header, ID, and label
    with open('submission.csv', 'w', encoding='utf-8') as file:
        file.write('ID,label')

        predicted_labels = []
        
        # make the model predict
        model.eval()
        for test_batch_sequences, _ in test_dataloader:
            outputs = model(test_batch_sequences) # batch_size * seq_length * output_size
            # Calculate accuracy
            batch_predicted_labels = outputs.argmax(dim=2)  # Get the index with the maximum probability
            mask = (test_batch_sequences != 0) & (test_batch_sequences != 2) & (test_batch_sequences != 8) & (test_batch_sequences != 15) & (test_batch_sequences != 16) & (test_batch_sequences != 26) & (test_batch_sequences != 40) & (test_batch_sequences != 43)
            batch_predicted_labels = batch_predicted_labels[mask]
            
            # extend these predictions to the predicted_labels list
            predicted_labels.extend(batch_predicted_labels.tolist())
            
        print('predicted_labels length: ', len(predicted_labels))
        
        # write the predictions to the file
        for i in range(len(predicted_labels)):
            file.write(f'\n{i},{predicted_labels[i]}')
            
def predict_single_sentence(model, original_sentence='', max_len=200, char_to_index={}, indicies_to_labels={}, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #; print(device)
    # preprocess the original sentence
    preprocessed_sentence = original_sentence.strip()
    preprocessed_sentence = clean([preprocessed_sentence])[0]
    
    # tokenize the sentence
    preprocessed_sentence = re.compile(r'[\n\r\t]').sub('', preprocessed_sentence)
    preprocessed_sentence = re.compile(r'\s+').sub(' ', preprocessed_sentence)
    preprocessed_sentence = preprocessed_sentence.strip()

    tokenized_sentences = []
    
    # split the line into sentences by dot
    dot_splitted_list = preprocessed_sentence.split('.')

    # remove last string if empty
    if dot_splitted_list[-1] == '':
        dot_splitted_list = dot_splitted_list[:-1]

    for dot_splitted in dot_splitted_list:
        dot_splitted = dot_splitted.strip()
        # Split the line into sentences of max_len, without cutting words
        sentences = textwrap.wrap(dot_splitted, max_len)

        for sentence in sentences:
            tokenized_sentences.append(sentence)
            
    sentence_sequences = convert2idx(data=tokenized_sentences, char_to_index=char_to_index, max_len=max_len, device=device)
    
    dataset = TensorDataset(sentence_sequences, sentence_sequences)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    predicted_labels = []
        
    # make the model predict
    model.eval()
    for batch_sequences, batch_labels in dataloader:
        outputs = model(batch_sequences) # batch_size * seq_length * output_size
        # Calculate accuracy
        batch_predicted_labels = outputs.argmax(dim=2)  # Get the index with the maximum probability
        mask = (batch_labels != 15) & (batch_sequences != 2) & (batch_sequences != 8) & (batch_sequences != 15) & (batch_sequences != 16) & (batch_sequences != 26) & (batch_sequences != 40) & (batch_sequences != 43)
        batch_predicted_labels = batch_predicted_labels[mask]
        
        # extend these predictions to the predicted_labels list
        predicted_labels.extend(batch_predicted_labels.tolist())
        
    predicted_sentence = ""
    predicted_char_index = 0
    for char in original_sentence:
        predicted_sentence += char
        # if the char is an unknown char, ., space, ?, ... or :, then keep it as it is
        if char not in char_to_index:
            continue
        elif char_to_index[char] == 2 or char_to_index[char] == 8 or char_to_index[char] == 15 or char_to_index[char] == 16 or char_to_index[char] == 26 or char_to_index[char] == 40 or char_to_index[char] == 43:
            continue
        # get it predicted diacritic (char or tuple of chars)
        predicted_class = indicies_to_labels[predicted_labels[predicted_char_index]]
        if type(predicted_class) is tuple:
            predicted_sentence += chr(predicted_class[0]) + chr(predicted_class[1])
            predicted_char_index += 1
        elif predicted_class == 0:
            # if the predicted diacritic is 0 (no diacritic), then keep the char as it is
            predicted_char_index += 1
        else:
            # else, add the predicted diacritic to the char
            predicted_sentence += chr(predicted_class)
            predicted_char_index += 1
                
    return predicted_sentence
            

# main
if __name__ == "__main__":
    # supress warnings
    warnings.filterwarnings('ignore')
    
    # NOTE: Uncomment this line to train the model
    # train()
    # load the model
    model = load_model(model_path)
    
    # predict whole test data, and output labels to submission.csv
    predict_test(model)
    
    # predict a single sentence
    test_sentence = "ليس للوكيل بالقبض أن يبرأ المدين أو يهب الدين له أو يأخذ رهنا من المدين في مقابل الدين أو يقبل إحالته على شخص آخر لكن له أن يأخذ كفيلا لكن ليس له أن يأخذ كفيلا بشرط براءة الأصيل انظر المادة ( 648 ) ( الأنقروي ، الطحطاوي وصرة الفتاوى ، البحر ) ."
    print(len(test_sentence))
    print(test_sentence)
    predicted_sentence = predict_single_sentence(model=model, original_sentence=test_sentence, max_len=max_len, char_to_index=char_to_index, indicies_to_labels=indicies_to_labels, batch_size=validation_batch_size)
    print(len(remove_diactrics([predicted_sentence])[0]))
    print(predicted_sentence)