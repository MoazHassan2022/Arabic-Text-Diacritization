import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(CharLSTM, self).__init__()
        # chars embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # LSTM layers
        # batch_first: it means that the input tensor has its first dimension representing the batch size
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # output layer
        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        embedded = self.embedding(x) # batch_size * seq_length * embedding_size
        lstm_out, _ = self.lstm(embedded) # batch_size * seq_length * hidden_size
        output = self.output(lstm_out)  # batch_size * seq_length * output_size
        return output