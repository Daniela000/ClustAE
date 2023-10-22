import torch.nn as nn
import torch.functional as F
import torch
from LSTM_AE import LSTMAutoencoder
torch.manual_seed(0)


class FSGAutoencoder(LSTMAutoencoder):
    def __init__(self, seq_len,dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head):
        super().__init__(seq_len, dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
        self.seq_len = seq_len
        self.hidden_size = dynamic_hidden_size
        #self.fsg_classifier = nn.Sequential(nn.Linear(dynamic_hidden_size,1))
        self.fsg_classifier = nn.Sequential(nn.Linear(dynamic_hidden_size, dynamic_hidden_size), nn.ReLU(), nn.Linear(dynamic_hidden_size, dynamic_hidden_size), nn.ReLU(), nn.Linear(dynamic_hidden_size, dynamic_hidden_size), nn.ReLU(),nn.Linear(dynamic_hidden_size, 1), nn.Sigmoid())
    def forward(self, dynamic_x):


        dynamic_x, dynamic_encoded = self.encoder(dynamic_x)
        #outcome = self.predictor(dynamic_encoded)
        fsg_classification = self.fsg_classifier(dynamic_encoded)
        #fsg_classification = torch.sigmoid(fsg_classification)
        dynamic_decoded = self.decoder(dynamic_encoded)
        
        return dynamic_decoded, fsg_classification.squeeze(1)
