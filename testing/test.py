# Import necessary libraries
import sys
import os
import pickle
sys.path.insert(0, os.path.abspath('..'))
from tokenization.tokenize import get_dataloader
from model.BLSTM.model import CharLSTM

# best_model path
model_path = 'best_model.pkl'

# these are the indices of the characters in the vocabulary
char_to_index = {'د': 1, '؟': 2, 'آ': 3, 'إ': 4, 'ؤ': 5, 'ط': 6, 'م': 7, '،': 8, 'ة': 9, 'ت': 10, 'ر': 11, 'ئ': 12, 'ا': 13, 'ض': 14, '!': 15, ' ': 16, 'ك': 17, 'غ': 18, 'س': 19, 'ص': 20, 'أ': 21, 'ل': 22, 'ف': 23, 'ظ': 24, 'ج': 25, '؛': 26, 'ن': 27, 'ع': 28, 'ب': 29, 'ث': 30, 'ه': 31, 'خ': 32, 'ى': 33, 'ء': 34, 'ز': 35, 'ق': 36, 'ي': 37, 'ش': 38, 'ح': 39, ':': 40, 'ذ': 41, 'و': 42}
index_to_char = {1: 'د', 2: '؟', 3: 'آ', 4: 'إ', 5: 'ؤ', 6: 'ط', 7: 'م', 8: '،', 9: 'ة', 10: 'ت', 11: 'ر', 12: 'ئ', 13: 'ا', 14: 'ض', 15: '!', 16: ' ', 17: 'ك', 18: 'غ', 19: 'س', 20: 'ص', 21: 'أ', 22: 'ل', 23: 'ف', 24: 'ظ', 25: 'ج', 26: '؛', 27: 'ن', 28: 'ع', 29: 'ب', 30: 'ث', 31: 'ه', 32: 'خ', 33: 'ى', 34: 'ء', 35: 'ز', 36: 'ق', 37: 'ي', 38: 'ش', 39: 'ح', 40: ':', 41: 'ذ', 42: 'و'}

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

def calc_accuracy():
    # max sentence length
    max_len = 200

    # batch size, number of sentences to be processed at once
    batch_size = 256

    # change this to the path of the dataset files
    dataset_path = '../dataset'

    test_dataloader = get_dataloader(data_type='test', max_len=max_len, batch_size=batch_size, dataset_path=dataset_path, char_to_index=char_to_index, labels=labels, with_labels=True)

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # test model
    correct_predictions = 0
    total_predictions = 0
    model.eval()
    for test_batch_sequences, test_batch_labels in test_dataloader:
        outputs = model(test_batch_sequences) # batch_size * seq_length * output_size
        # Calculate accuracy
        predicted_labels = outputs.argmax(dim=2)  # Get the index with the maximum probability
        mask = (test_batch_labels != 15) & (test_batch_sequences != 2) & (test_batch_sequences != 8) & (test_batch_sequences != 16) & (test_batch_sequences != 26) & (test_batch_sequences != 40)
        correct_predictions += ((predicted_labels == test_batch_labels) & mask).sum().item()
        total_predictions += mask.sum().item()

    accuracy = correct_predictions / total_predictions

    print(f'After reading, Accuracy: {accuracy * 100:.3f}%')