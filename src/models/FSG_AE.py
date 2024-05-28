import torch.nn as nn
import torch.functional as F
import torch
from models.Temp_Static_AE import STAutoencoder
torch.manual_seed(0)


class FSGAutoencoder(STAutoencoder):
    def __init__(self, seq_len,dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head,static_input_size ):
        super().__init__(seq_len, dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head, static_input_size)
        self.seq_len = seq_len
        self.hidden_size = dynamic_hidden_size
        #self.fsg_classifier = nn.Sequential(nn.Linear(dynamic_hidden_size,1))
        self.fsg_classifier = nn.Sequential(nn.Linear(dynamic_hidden_size, dynamic_hidden_size), nn.ReLU(), nn.Linear(dynamic_hidden_size, dynamic_hidden_size), nn.ReLU(), nn.Linear(dynamic_hidden_size, dynamic_hidden_size), nn.ReLU(),nn.Linear(dynamic_hidden_size, 1), nn.Sigmoid())
    def forward(self, dynamic_x, static_x):


        static_encoded, dynamic_encoded,dynamic_x = self.encoder(dynamic_x,static_x)
        #outcome = self.predictor(dynamic_encoded)
        fsg_classification = self.fsg_classifier(dynamic_encoded)
        #fsg_classification = torch.sigmoid(fsg_classification)
        static_decoded, dynamic_decoded = self.decoder(dynamic_encoded, static_encoded)
        
        return dynamic_decoded, static_decoded,fsg_classification.squeeze(1)
