import torch.nn as nn
import torch.functional as F
import torch
from models.LSTM_AE import LSTMEncoder, LSTMDecoder
torch.manual_seed(0)


class STEncoder(LSTMEncoder):
    def __init__(self, seq_len, dynamic_input_size, hidden_size, num_layers, bidirectional1,bidirectional2,num_head, static_input_size):
        super().__init__( seq_len, dynamic_input_size, hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.dynamic_input_size=dynamic_input_size

        self.static_encoder = nn.Sequential(   
            nn.Conv1d(in_channels=static_input_size, out_channels=2*static_input_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*static_input_size, out_channels=4*static_input_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4*static_input_size, out_channels=4*static_input_size, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(4*static_input_size, hidden_size)
        )
        self.static_encoder_attention = nn.MultiheadAttention(static_input_size, num_heads=1)
        self.norm1 = nn.LayerNorm(static_input_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)     

        # Initialize the weights of the LSTM layers using He initialization

        nn.init.xavier_uniform_(self.static_encoder_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.static_encoder_attention.out_proj.weight)
        if self.static_encoder_attention.bias_k is not None:
            nn.init.constant_(self.static_encoder_attention.bias_k, 0)


    def forward(self, dynamic_x, static_x):
        dynamic_x, hidden_n = super().forward(dynamic_x)
        #static_x= static_x.unsqueeze(1)
        #print(static_x)
        #if static_x.shape[0] == 1: 
            #static_x = static_x.repeat(2,1,1)
        static_x,_ = self.static_encoder_attention(static_x, static_x, static_x)
        #print(static_x.shape)
        #static_x = static_x + self.dropout(static_x)
        static_x = self.norm1(static_x)
        
        static_x= static_x.unsqueeze(1)
        #if static_x.shape[0] == 1: 
            #static_x = static_x.repeat(2,1,1)
        #print(static_x.shape)
        #static_x= self.static_encoder(static_x)
        static_x = self.static_encoder(static_x.permute(0,2,1)).permute(0,1)
        #static_x = static_x + self.dropout(static_x)
        #static_x = self.norm2(static_x)
          
        return static_x,hidden_n, dynamic_x

         
class STDecoder(LSTMDecoder):
    def __init__(self, seq_len, dynamic_input_size, hidden_size, num_layers, bidirectional1,bidirectional2,num_head, static_input_size):
        super().__init__(seq_len, dynamic_input_size, hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.norm1 = nn.LayerNorm(static_input_size)
        self.fc = nn.Linear(hidden_size,4*static_input_size)
        self.static_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=4*static_input_size, out_channels=4*static_input_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4*static_input_size, out_channels=2*static_input_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2*static_input_size, out_channels=static_input_size, kernel_size=3, padding=1))
    
        self.static_decoder_attention = nn.MultiheadAttention(embed_dim=static_input_size, num_heads=1)


        nn.init.xavier_uniform_(self.static_decoder_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.static_decoder_attention.out_proj.weight)
        if self.static_decoder_attention.bias_k is not None:
            nn.init.constant_(self.static_decoder_attention.bias_k, 0)        


    def forward(self, dynamic_x, static_x):
        dynamic_x = super().forward(dynamic_x)
        if static_x.dim() == 2: 
            static_x = static_x.unsqueeze(1)
        static_x = self.fc(static_x)
        #print(static_x.shape)
        static_x = static_x.reshape(static_x.shape[0],-1,static_x.shape[1])

        #static_x = self.static_decoder(static_x.permute(0,2,1)).permute(0,2,1)
        static_x = self.static_decoder(static_x).permute(0,2,1)
        static_x, _ = self.static_decoder_attention(static_x,static_x,static_x)
        static_x = self.norm1(static_x)
        static_x = static_x.squeeze(1) 
        #print(static_x.shape)

        return static_x, dynamic_x


class STAutoencoder(nn.Module):
    def __init__(self, seq_len,dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head, static_input_size):
        super(STAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = dynamic_hidden_size
        
        #for dynamic features
        self.encoder = STEncoder(seq_len, dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head, static_input_size)
        self.decoder = STDecoder(seq_len, dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head, static_input_size)
        
    def forward(self, dynamic_x, static_x):

        #static_noise = torch.randn_like(static_x) * 0.1
        #static_x = static_x + static_noise

        static_encoded, dynamic_encoded, _ = self.encoder(dynamic_x, static_x)
        static_decoded, dynamic_decoded = self.decoder(dynamic_encoded, static_encoded)

        return dynamic_decoded, static_decoded
