from re import S
import torch
from models.LSTM_AE import LSTMAutoencoder
from models.Temp_Static_AE import STAutoencoder
from models.FSG_AE import FSGAutoencoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import preprocessing.constants as constants
import random

def transform_data(temp_data, static_data):
    scaler = StandardScaler()
    temp_data = temp_data.copy()
    static_data = static_data.copy()
    y_true = temp_data['Evolution'].values
    refs = temp_data['Patient_ID'].values
    temp_data.drop(['Patient_ID', 'Evolution'], axis=1, inplace = True)
    temp_data = pd.DataFrame(scaler.fit_transform(temp_data), columns = temp_data.columns)
    dynamic_data = []
    for idx, row in temp_data.iterrows(): 
        time_points = []
        for i in range(constants.MIN_APP):
            features = [str(i) + item for item in list(constants.TEMPORAL_FEATURES.keys())]
            time_points.append(row[features].values.tolist())
        dynamic_data.append(time_points)
    static_features = list(constants.STATIC_FEATURES.keys())

    if static_features != []:
        static_data =  pd.DataFrame(scaler.fit_transform(static_data[static_features]), columns = static_features)
        #[static_features] = scaler.fit_transform(static_data[static_features])

    static_data = static_data[static_features].values.tolist()
    return refs, y_true, dynamic_data, static_data

def generate_fsg(train_data, static_train_set):
    #create a y that is 1 if record true and 0 otherwise
    y_true = [1] * len(train_data)
    new_train_data = train_data.copy()
    static_train_set = static_train_set*2
    for record in train_data:
        #shuffle lists
        new_record = record.copy()
        random.shuffle(new_record)
        new_train_data.append(new_record)
        y_true.append(0)
        
    #shuffle train_data and y
    zipped = list(zip(new_train_data,static_train_set, y_true))
    random.shuffle(zipped)
    new_train_data, static_train_set, y_true = zip(*zipped)
    return new_train_data,static_train_set, y_true

def train(static_train_set, static_val_set, dynamic_train_set, dynamic_val_set, type_model):
    dynamic_features = list(constants.TEMPORAL_FEATURES.keys())
    static_features = list(constants.STATIC_FEATURES.keys())

    n_features = len(dynamic_features)
    n_static_features = len(static_features)
    hidden_size = 4 #2
    num_layers = 1
    seq_len = constants.MIN_APP

    patience = 2
    trigger_times = 0

    bidirectional1 = True 
    bidirectional2 = False
    num_head = 4

    lr = 1e-4
    
    # Model Initialization
    if type_model == 'simple':
        model = LSTMAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
        output_folder = 'simple_AE_test'
    elif type_model == 'temp_static':
        model = STAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head,n_static_features)
        output_folder = 'temp_static_AE_test'
        alpha = 0.6
        lr = 1e-3
    #elif type_model == 'predictor':
        #model = PredAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head,n_static_features)
        #output_folder = 'predictor_model'
        #alpha = 0.5
        #outcome_loss = torch.nn.BCELoss()
    elif type_model == 'fsg':
        dynamic_train_set, static_train_set, train_fsg_true = generate_fsg(dynamic_train_set, static_train_set)
        dynamic_val_set, static_val_set, val_fsg_true = generate_fsg(dynamic_val_set, static_val_set)
        output_folder = 'fsg_model_test'
        model = FSGAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head, n_static_features)
        alpha = 0.5
        #outcome_loss = torch.nn.BCELoss()
        class_loss =torch.nn.BCELoss()

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = lr, weight_decay=0.001)
    epochs = 200
    history = dict(train=[], val=[])
    batch_size = 2

    for epoch in range(epochs):
        model = model.train()
        train_losses = []
        for i in range(0, len(dynamic_train_set), batch_size):
            dynamic_record = dynamic_train_set[i:i+batch_size]                      
            dynamic_record = torch.tensor(dynamic_record)
            if type_model == 'temp_static' or type_model == 'fsg':
                static_record = static_train_set[i:i+batch_size]
                static_record = torch.tensor(static_record)   

            # The gradients are set to zero           
            optimizer.zero_grad()
            # Output of Autoencoder
            if type_model == 'simple':
                reconstructed = model(dynamic_record) 
                loss = loss_function(reconstructed,dynamic_record)
 
            elif type_model == 'temp_static':
                dynamic_reconstructed, static_reconstructed  = model(dynamic_record,static_record)
                dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                static_loss = loss_function(static_reconstructed,static_record)
                loss = alpha*dynamic_loss + (1-alpha)*static_loss
            #elif type_model == 'predictor':
                #y_true = torch.tensor(train_y_true[i:i+batch_size], dtype = torch.float32)
                #dynamic_reconstructed, static_reconstructed, outcome  = model(dynamic_record,static_record)
                #dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                #static_loss = loss_function(static_reconstructed,static_record)
                #loss = 0.05*(alpha*dynamic_loss + (1-alpha)*static_loss) +  0.95*outcome_loss(outcome, y_true)
            elif type_model == 'fsg':
                #y_true = torch.tensor(y_true[i:i+batch_size], dtype = torch.float32)
                fsg_true = torch.tensor(train_fsg_true[i:i+batch_size], dtype = torch.float32)
                dynamic_reconstructed, static_reconstructed, fsg  = model(dynamic_record,static_record)
                dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                static_loss = loss_function(static_reconstructed,static_record)
                loss = (alpha*dynamic_loss + (1-alpha)*static_loss) +  class_loss(fsg,fsg_true)
                loss = 0.95*dynamic_loss + 0.05*class_loss(fsg,fsg_true)
                    
            
            loss.backward()
            # .step() performs parameter update
            optimizer.step()                     
            # Storing the losses in a list for plotting
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for i in range(0, len(dynamic_val_set), batch_size):
                dynamic_record = dynamic_val_set[i:i+batch_size] 

                dynamic_record = torch.tensor(dynamic_record)
                if type_model == 'temp_static':
                    static_record = static_val_set[i:i+batch_size]
                    static_record = torch.tensor(static_record)   

                # Output of Autoencoder
                if type_model == 'simple':
                    reconstructed = model(dynamic_record) 
                    loss = loss_function(reconstructed,dynamic_record)
    
                elif type_model == 'temp_static':
                    dynamic_reconstructed, static_reconstructed  = model(dynamic_record,static_record)
                    dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                    static_loss = loss_function(static_reconstructed,static_record)
                    loss = alpha*dynamic_loss + (1-alpha)*static_loss

                elif type_model == 'fsg':
                    fsg_true = torch.tensor(val_fsg_true[i:i+batch_size], dtype = torch.float32)
                    dynamic_reconstructed, static_reconstructed, fsg  = model(dynamic_record,static_record)
                    dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                    static_loss = loss_function(static_reconstructed,static_record)
                    loss = (alpha*dynamic_loss + (1-alpha)*static_loss) +  class_loss(fsg,fsg_true)
                    loss = 0.95*dynamic_loss + 0.05*class_loss(fsg,fsg_true)

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        #early stop
        if epoch > 1 and history['val'][-1] < 0.0001 + val_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
        else:
            trigger_times = 0
        if trigger_times >= patience:
            print('Early stopping!')
            break
            
            
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    EPOCH = epoch
    #PATH = type_model + "turim_niv_180_5app.pt"
    PATH = type_model + '_turim_no_nan_val.tjm'
    LOSS = train_loss
    torch.save(model, PATH)
    #torch.save({
                #'epoch': EPOCH,
                #'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                #'loss': LOSS,
                #}, PATH)
    
    return model

def get_encoded_reps(model, dynamic_data, static_data, type_model):
    model = model.eval()
    if type_model == 'simple':
        _, reps = model.encoder(torch.tensor(dynamic_data))
    else:
        static_reps,dynamic_reps, _= model.encoder(torch.tensor(dynamic_data), torch.tensor(static_data))
        #print(static_reps.shape)
        #print(dynamic_reps.shape)
        static_reps = static_reps.squeeze(1)
        reps = torch.cat((static_reps,dynamic_reps), dim =1)
    reps = reps.detach().numpy()
    return reps

def finetune_model(model, train_group, test_group):
    batch_size = 2
    trigger_times = 0
    history = dict(train=[], val=[])
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-4, weight_decay=0.001)
    #finetune
    for epoch in range(100):
        model = model.train()
        train_losses = []
        for i in range(0, len(train_group), batch_size):
            dynamic_record = train_group[i:i+batch_size]                      
            dynamic_record = torch.tensor(dynamic_record)
            
            # The gradients are set to zero           
            optimizer.zero_grad()
            # Output of Autoencoder

            reconstructed = model(dynamic_record) 
            loss = loss_function(reconstructed,dynamic_record)

            loss.backward()
            # .step() performs parameter update
            optimizer.step()                     
            # Storing the losses in a list for plotting
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for i in range(0, len(test_group), batch_size):
                dynamic_record = test_group[i:i+batch_size] 

                dynamic_record = torch.tensor(dynamic_record)
                reconstructed = model(dynamic_record) 
                loss = loss_function(reconstructed,dynamic_record)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        #early stop
        if epoch > 1 and history['val'][-1] < 0.0001 + val_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
        else:
            trigger_times = 0
        if trigger_times >= 3:
            print('Early stopping!')
            break
        
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    return model