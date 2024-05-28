from flask import Flask, render_template, request, jsonify, session
import torch
import sys
from sklearn.preprocessing import StandardScaler
import joblib
import os
from joblib import load
import numpy as np

app = Flask(__name__, static_folder='frontend/static', template_folder = 'frontend/templates')
app.secret_key = 'clustAE001!'

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for selecting the model
@app.route('/model', methods=['GET'])
def select_model():
    return render_template('questions.html')

@app.route('/get_question_data', methods=['POST'])
def get_question_data():
    #get data
    model = request.args.get('model')
    data = request.form

    #compute and save the ALSFRS-R scores
    input = compute_scores(data)
    session['scores'] = input

    #if model is clustAE predict the cluster, if model is STClustAE render static form page
    if model == 'TClustAE':
        norm_data = normalize(input, model)
        input_tensor = torch.tensor(norm_data, dtype = torch.float32)
        input_tensor = input_tensor.unsqueeze(0)

        model = torch.load('simple_turim_no_nan_val.pt')
        model.eval()
        _, reps = model.encoder(input_tensor)
        reps = reps.detach().numpy()

        classifier = joblib.load('simple_nearest_centroids_original.joblib')
        
        #_, reps = model.encoder(input_tensor)
        label = classifier.predict(reps)
        label = label[0]
        print(label)
        #render the results page
        if label == 0:
            return render_template('tclustae_results/mps.html') # page with the MPs description
        elif label == 1:
            return render_template('tclustae_results/sp.html') # page with the SP description
        elif label == 2:
            return render_template('tclustae_results/mpr.html') # page with the MPr description
        elif label == 3:
            return render_template('tclustae_results/mpb.html') # page with the MPb description
        else:
            return f'ERROR: Selected model {model}' # error
    else:
        print('ola1')
        return render_template('static_data.html') # page to add the static info  
    
@app.route('/get_static_data', methods=['POST'])
def get_static_data():
    #get data
    print('ola2')
    static_data = request.form
    static_data = [static_data['DiagnosticDelay'], static_data['Age_onset'], static_data['Onset'],static_data['Gender'], static_data['BMI'],
                   static_data['C9orf72'],static_data['UMNvsLMN']]
    static_data = np.array(static_data)
    scores = session.get('scores')

    #normalize data
    scores, static_data = normalize(scores, 'STClustAE', static_data)
    input_tensor = torch.tensor(scores, dtype = torch.float32)
    input_tensor = input_tensor.unsqueeze(0)

    #load model and compute the encoded reps
    model = torch.load('temp_static_turim_no_nan_val.pt')
    model.eval()
    static_reps,dynamic_reps, _= model.encoder(torch.tensor(input_tensor), torch.tensor(static_data))
    static_reps = static_reps.squeeze(1)
    reps = torch.cat((static_reps,dynamic_reps), dim =1)
    reps = reps.detach().numpy()

    # Load the classifier and predict the cluster
    classifier = joblib.load('temp_static_nearest_centroids_original.joblib')
    label = classifier.predict(reps)

    print(label)
    #render the results page
    if label == 0:
        return render_template('stclustae_results/sp.html') # page with the MPs description
    elif label == 1:
        return render_template('stclustae_results/fp.html') # page with the SP description
    elif label == 2:
        return render_template('stclustae_results/mpb.html') # page with the MPr description
    elif label == 3:
        return render_template('stclustae_results/mp.html') # page with the MPb description
    else:
        return f'ERROR: Selected model {model}' # error
    
def mitos_stage(data, i):
    mov = 1 if int(data[f'Q8_{i}']) < 4 or int(data[f'Q6_{i}']) < 4 else 0
    swa = 1 if int(data[f'Q3_{i}']) < 4 else 0
    com = 1 if int(data[f'Q1_{i}']) < 4 and int(data[f'Q4_{i}']) < 4 else 0
    bre = 1 if int(data[f'Q10_{i}']) < 4 or int(data[f'Q12_{i}']) < 4 else 0

    return mov + swa + com + bre


def compute_scores(data):
    questions = []
    for i in range(1,4):
        alsfrsrul = int(data[f'Q4_{i}']) + int(data[f'Q5_{i}'])
        alsfrsrll = int(data[f'Q8_{i}']) + int(data[f'Q9_{i}'])
        alsfrsrt = int(data[f'Q6_{i}']) + int(data[f'Q7_{i}'])
        alsfsrb = int(data[f'Q1_{i}']) + int(data[f'Q2_{i}'])  + int(data[f'Q3_{i}'])
        alsfrsr =  int(data[f'Q10_{i}']) + int(data[f'Q11_{i}']) + int(data[f'Q12_{i}'])
        total = alsfrsrul + alsfrsrll + alsfrsrt + alsfsrb + alsfrsr
        mitos = mitos_stage(data, i)

        questions.append(alsfrsrul)
        questions.append(alsfrsrll)
        questions.append(alsfrsrt)
        questions.append(alsfsrb)
        questions.append(alsfrsr)
        questions.append(total)
        questions.append(mitos)
    return questions

def normalize(input, model, static_data = []):
    
    if model == 'TClustAE':
        scaler = load('tclustae_scaler.joblib')
        input = np.array(input)
        input = scaler.transform(input.reshape(1,-1))
        input = input.reshape(3, 7).tolist()
        print(input)
        
        return input
    if model == 'STClustAE':
        temp_scaler = load('stclustae_temp_scaler.joblib')
        input = np.array(input)
        input = temp_scaler.transform(input.reshape(1,-1))
        input = input.reshape(3, 7).tolist()

        static_scaler = load('stclustae_static_scaler.joblib')
        static_data = static_scaler.transform(static_data.reshape(1,-1)).tolist()
        return input, static_data     

if __name__ == '__main__':
    app.run(debug=True)