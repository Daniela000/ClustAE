from flask import Flask, render_template, request, jsonify, session
import torch
import sys
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__, template_folder = 'templates')
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

        #model = LSTMAutoencoder()
        #model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simple_turim_no_nan_val.pt'))
        model = torch.load('simple_turim_no_nan_val.tjm')
        model.eval()

        # Load the classifier and assign a cluster
        #classifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simple_nearest_centroids_original.joblib'))
        classifier = joblib.load('simple_nearest_centroids_original.joblib')

        # Ensure the input tensor is of type float and is on the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = torch.tensor(norm_data, dtype=torch.float).to(device)

        # Forward pass through the encoder
        with torch.no_grad():
            _, reps = model.encoder(input_tensor)
        #_, reps = model.encoder(input_tensor)
        label = classifier.predict(reps.cpu().numpy())

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
        return render_template('static_data.html') # page to add the static info  
    
@app.route('/get_static_data', methods=['POST'])
def get_static_data():
    #get data
    static_data = request.form
    scores = session.get('scores')

    #normalize data
    scores, static_data = normalize(scores, 'STClustAE', static_data)

    #load model and compute the encoded reps
    model = torch.jit.load('temp_static_turim_no_nan_val.pt')
    model.eval()
    static_reps,dynamic_reps, _= model.encoder(torch.tensor(scores), torch.tensor(static_data))
    static_reps = static_reps.squeeze(1)
    reps = torch.cat((static_reps,dynamic_reps), dim =1)

    # Load the classifier and predict the cluster
    #classifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simple_nearest_centroids_original.joblib'))
    classifier = joblib.load('temp_static_nearest_centroids_original.joblib')
    label = classifier.predict(reps.cpu().numpy())

    #render the results page
    if label == 0:
        return render_template('stclustae_results/mps.html') # page with the MPs description
    elif label == 1:
        return render_template('stclustae_results/sp.html') # page with the SP description
    elif label == 2:
        return render_template('stclustae_results/mpr.html') # page with the MPr description
    elif label == 3:
        return render_template('stclustae_results/mpb.html') # page with the MPb description
    else:
        return f'ERROR: Selected model {model}' # error

def compute_scores(data):
    questions = []
    for i in range(1,4):
        alsfrsrul = data[f'Q4_{i}'] + data[f'Q5_{i}']
        alsfrsrll = data[f'Q8_{i}'] + data[f'Q9_{i}']
        alsfrsrt = data[f'Q6_{i}'] + data[f'Q7_{i}']
        alsfsrb = data[f'Q1_{i}'] + data[f'Q2_{i}']  + data[f'Q3_{i}']
        alsfrsr =  data[f'Q10_{i}'] + data[f'Q11_{i}'] + data[f'Q12_{i}']
        total = alsfrsrul + alsfrsrll + alsfrsrt + alsfsrb + alsfrsr
        questions.append([alsfrsrul, alsfrsrll, alsfrsrt, alsfsrb, alsfrsr, total])
    return questions

def normalize(input, model, static_data = []):
    scaler = StandardScaler()
    input = scaler.fit_transform(input)
    if model == 'TClustAE':
        return input
    #if model == 'STClustAE':
        #call label encoder for categorical features (gender, umnvslmn, c90rf72,onset)
        #transform
        #static_data = scaler.fit_transform(input)
        #return input, static_data
        




    
if __name__ == '__main__':
    app.run(debug=True)