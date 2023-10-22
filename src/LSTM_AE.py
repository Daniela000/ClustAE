import torch.nn as nn
import torch.functional as F
import torch
torch.manual_seed(0)


class LSTMEncoder(nn.Module):
    def __init__(self, seq_len, dynamic_input_size, hidden_size, num_layers,bidirectional1,bidirectional2,num_head):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.dynamic_input_size=dynamic_input_size
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lstm1 = nn.LSTM(dynamic_input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional1)
        if bidirectional1:
            self.lstm2 = nn.LSTM(2*hidden_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional2)   
        else:
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional2)

        if bidirectional2:
            if 2*hidden_size % num_head == 0:
                self.encoder_attention = nn.MultiheadAttention(embed_dim=2*hidden_size, num_heads = num_head)
            else:
                self.encoder_attention = nn.MultiheadAttention(embed_dim=2*hidden_size, num_heads = 1)
            self.encoder_out = nn.Linear(2*hidden_size, hidden_size)

        else:
            self.encoder_out = nn.Linear(hidden_size, hidden_size)
            if hidden_size % num_head == 0:
                    self.encoder_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads = num_head)
            else:
                self.encoder_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads = 1)
            
            
        
        #self.dropout_layer = torch.nn.Dropout(p=0.1)
        
        
        # Initialize the weights of the LSTM layers using He initialization
        for layer in [self.lstm1,self.lstm2]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='tanh')
                if 'bias' in name:
                    nn.init.constant_(param, 0.01)
        nn.init.kaiming_uniform_(self.encoder_out.weight)
        nn.init.constant_(self.encoder_out.bias, 0.01)

        nn.init.uniform_(self.encoder_attention.in_proj_weight)
        nn.init.uniform_(self.encoder_attention.out_proj.weight)
        if self.encoder_attention.bias_k is not None:
            nn.init.constant_(self.encoder_attention.bias_k, 0.01)
        

    def forward(self, dynamic_x):           
        #print(dynamic_x.shape)
        batch_size = dynamic_x.shape[0]
        #print(dynamic_x)
        #dynamic_x = dynamic_x.reshape((batch_size, self.seq_len, self.dynamic_input_size))      
        
        dynamic_x, (hidden_n,_)= self.lstm1(dynamic_x)
        #print(dynamic_x)
        dynamic_x = self.relu(dynamic_x)
        dynamic_x, (hidden_n,_) = self.lstm2(dynamic_x)
        hidden_n = hidden_n[-1].reshape((batch_size, self.hidden_size))
        #hidden_n, _ = self.encoder_attention(hidden_n, hidden_n, hidden_n)
        
        #hidden_n = self.encoder_out(hidden_n)
   
        return dynamic_x, hidden_n

         
class LSTMDecoder(nn.Module):
    def __init__(self, seq_len, dynamic_input_size, hidden_size, num_layers,bidirectional1,bidirectional2,num_head):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len


        self.lstm1 = nn.LSTM(hidden_size,hidden_size, num_layers, batch_first=True,bidirectional = bidirectional2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if bidirectional2:
            self.lstm2 = nn.LSTM(2*hidden_size,hidden_size, num_layers, batch_first=True, bidirectional = bidirectional1)
        else:        
            self.lstm2 = nn.LSTM(hidden_size,hidden_size, num_layers, batch_first=True, bidirectional = bidirectional1)

        if bidirectional1: 
            if 2*hidden_size % num_head == 0:
                self.decoder_attention = nn.MultiheadAttention(embed_dim=2*hidden_size , num_heads=num_head)
            else:
                self.decoder_attention = nn.MultiheadAttention(embed_dim=2*hidden_size , num_heads=1)
            
            self.output_layer = nn.Linear(2*hidden_size, dynamic_input_size)
        else:
            if hidden_size % num_head == 0:
                self.decoder_attention = nn.MultiheadAttention(embed_dim=hidden_size , num_heads=num_head)
            else:
                self.decoder_attention = nn.MultiheadAttention(embed_dim=hidden_size , num_heads=1)
            self.output_layer = nn.Linear(hidden_size, dynamic_input_size)
        # Initialize the weights of the LSTM layers using He initialization
        for layer in [self.lstm1,self.lstm2]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='tanh')
                if 'bias' in name:
                    nn.init.constant_(param, 0.01)
        #nn.init.kaiming_uniform_(self.fc.weight)
        #nn.init.constant_(self.fc.bias, 0.01)

        nn.init.kaiming_uniform_(self.decoder_attention.in_proj_weight)
        nn.init.kaiming_uniform_(self.decoder_attention.out_proj.weight)
        if self.decoder_attention.bias_k is not None:
            nn.init.constant_(self.decoder_attention.bias_k, 0.01)      

        nn.init.kaiming_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.01)  
        

    def forward(self, hidden):
        batch_size = hidden.shape[0]
        hidden = hidden.unsqueeze(0)
        hidden = hidden.repeat(1,self.seq_len,1) 
        hidden = hidden.reshape((batch_size, self.seq_len, self.hidden_size)) 
        
        dynamic_x, _ = self.lstm1(hidden)
        dynamic_x = self.relu(dynamic_x)
        
        dynamic_x, _ = self.lstm2(dynamic_x)
        #dynamic_x, _ = self.decoder_attention(dynamic_x,dynamic_x,dynamic_x )
        dynamic_x= self.output_layer(dynamic_x)
        
        return dynamic_x


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = dynamic_hidden_size
        
        #for dynamic features
        self.encoder = LSTMEncoder(seq_len, dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
        #self.outcome_layer = nn.Linear(dynamic_hidden_size, 1)
        self.decoder = LSTMDecoder(seq_len,dynamic_input_size, dynamic_hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
        
    def forward(self, dynamic_x):

        # Add Gaussian noise with mean 0 and standard deviation 0.1
        #if self.training:
           # mean = 0
            #stddev = 0.001
            #noise = torch.randn(dynamic_x.size()) * stddev + mean
            #dynamic_x = dynamic_x + noise
        
        #new_dynamic_x =torch.zeros_like(dynamic_x)

        #for i, row in enumerate(dynamic_x):
            #noise = torch.randn(row.size()) * stddev + mean
            #record = row + noise
            #new_dynamic_x[i] = record

        #new_dynamic_x =torch.zeros_like(dynamic_x)
        _, hidden = self.encoder(dynamic_x)
        decoded = self.decoder(hidden)

        #return outcome.squeeze(1), decoded
        return decoded
    
