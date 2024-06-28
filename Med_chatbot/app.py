# Import additional libraries
from flask import Flask, render_template, request
import joblib
import numpy as np
from lime import lime_tabular
import pandas as pd


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template, request
import time


app = Flask(__name__)



path = "Mohammed-Altaf/Medical-ChatBot"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(path).to(device)

prompt_input = (
    "The conversation between human and AI assistant.\n"
    "[|Human|] {input}\n"
    "[|AI|]"
)

# Load the heart disease model
heart_model_filename = "logistic_regression_model.joblib"

diabetes_model = joblib.load("diabetes.joblib")
diabetes_X_train = pd.read_csv("diabetes_train.csv")
diabetes_feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
try:
    heart_model = joblib.load(heart_model_filename)
except Exception as e:
    print(f"Error loading the heart disease model: {e}")
    raise

heart_feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# Load training data for heart disease from CSV
try:
    heart_X_train = pd.read_csv("heart_X_train.csv")
except FileNotFoundError:
    print("heart_X_train.csv not found. Please check the file path.")
    raise

# Define Lime interpretation function for heart disease
def heart_lime_interpretation(features):
    heart_explainer = lime_tabular.LimeTabularExplainer(
        heart_X_train.values,  # Use heart_X_train as a pandas DataFrame
        feature_names=heart_feature_names,
        class_names=['No Heart Disease', 'Heart Disease'],
        discretize_continuous=True
    )

    # Convert the input features to a NumPy array
    heart_features_array = np.array(features, dtype=float)

    heart_explanation = heart_explainer.explain_instance(
        heart_features_array,
        heart_model.predict_proba,
        num_features=len(heart_features_array)
    )

    return heart_explainer, heart_explanation

# Function to convert Heart Lime explanation to HTML format
def heart_lime_to_html(heart_explanation):
    return heart_explanation.as_html()


lung_model = joblib.load("lungs_model.joblib")


lung_feature_names = ['YELLOW_FINGERS',
 'ANXIETY',
 'PEER_PRESSURE',
 'CHRONIC DISEASE',
 'FATIGUE ',
 'ALLERGY ',
 'WHEEZING',
 'ALCOHOL CONSUMING',
 'COUGHING',
 'SWALLOWING DIFFICULTY',
 'CHEST PAIN',
 'ANXYELFIN']

# Load training data for lung disease from CSV
try:
    lung_X_train = pd.read_csv("lungs_train.csv")
except FileNotFoundError:
    print("lung_X_train.csv not found. Please check the file path.")
    raise

# Define Lime interpretation function for lung disease
def lung_lime_interpretation(features):
    lung_explainer = lime_tabular.LimeTabularExplainer(
        lung_X_train.values,  # Use lung_X_train as a pandas DataFrame
        feature_names=lung_feature_names,
        class_names=['No Lung Disease', 'Lung Disease'],
        discretize_continuous=True
    )

    # Convert the input features to a NumPy array
    lung_features_array = np.array(features, dtype=float)

    lung_explanation = lung_explainer.explain_instance(
        lung_features_array,
        lung_model.predict_proba,
        num_features=len(lung_features_array)
    )

    return lung_explainer, lung_explanation

# Function to convert Lung Lime explanation to HTML format
def lung_lime_to_html(lung_explanation):
    return lung_explanation.as_html()

def diabetes_lime_interpretation(features):
    diabetes_explainer = lime_tabular.LimeTabularExplainer(
        diabetes_X_train.values,  # Use diabetes_X_train as a pandas DataFrame
        feature_names=diabetes_feature_names,
        class_names=['No Diabetes', 'Diabetes'],
        discretize_continuous=True
    )

    # Convert the input features to a NumPy array
    diabetes_features_array = np.array(features, dtype=float)

    diabetes_explanation = diabetes_explainer.explain_instance(
        diabetes_features_array,
        diabetes_model.predict_proba,
        num_features=len(diabetes_features_array)
    )

    return diabetes_explainer, diabetes_explanation

# Function to convert Diabetes Lime explanation to HTML format
def diabetes_lime_to_html(diabetes_explanation):
    return diabetes_explanation.as_html()

# Render the main page with the input form
@app.route('/')
def index():
    return render_template('index.html')

# Predict heart disease based on input and display the result with Lime interpretation
@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
    # Get input values from the form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['chest_pain_type']),
            float(request.form['resting_bp']),
            float(request.form['cholesterol']),
            float(request.form['fasting_bs']),
            float(request.form['resting_ecg']),
            float(request.form['max_hr']),
            float(request.form['exercise_angina']),
            float(request.form['oldpeak']),
            float(request.form['st_slope'])
        ]

        if not hasattr(heart_model, 'predict_proba') or not callable(getattr(heart_model, 'predict_proba', None)):
            raise AttributeError("The loaded heart disease model does not have a 'predict_proba' method.")

        # Make a heart disease prediction
        heart_raw_prediction = heart_model.predict_proba([features])[0]
        heart_probability_positive_class = heart_raw_prediction[1]  # Assuming 1 corresponds to the positive class

        # Get Heart Lime interpretation
        heart_explainer, heart_lime_explanation = heart_lime_interpretation(features)

        # Convert Heart Lime explanation to HTML format
        heart_lime_html = heart_lime_to_html(heart_lime_explanation)

        # Save Heart Lime explanation to HTML file
        heart_html_filename = 'static/heart_lime_visualization.html'
        with open(heart_html_filename, 'w', encoding='utf-8') as file:
            file.write(heart_lime_html)

        # Display the result in a popup
        heart_result = "Heart Disease" if heart_probability_positive_class >= 0.5 else "No Heart Disease"

        return render_template('result.html', result=heart_result, lime_visualization_path=heart_html_filename)
    return render_template('index.html')

@app.route('/lung_disease', methods=['GET', 'POST'])
def lung_disease():
    if request.method == 'POST':
        # Get input values from the form for lung disease prediction
        lung_features = [
            float(request.form['YELLOW_FINGERS']),
            float(request.form['ANXIETY']),
            float(request.form['PEER_PRESSURE']),
            float(request.form['CHRONIC_DISEASE']),
            float(request.form['FATIGUE']),
            float(request.form['ALLERGY']),
            float(request.form['WHEEZING']),
            float(request.form['ALCOHOL_CONSUMING']),
            float(request.form['COUGHING']),
            float(request.form['SWALLOWING_DIFFICULTY']),
            float(request.form['CHEST_PAIN']),
            float(request.form['ANXYELFIN'])
        ]

        if not hasattr(lung_model, 'predict_proba') or not callable(getattr(lung_model, 'predict_proba', None)):
            raise AttributeError("The loaded lung disease model does not have a 'predict_proba' method.")

        # Make a lung disease prediction
        lung_raw_prediction = lung_model.predict_proba([lung_features])[0]
        lung_probability_positive_class = lung_raw_prediction[1]  # Assuming 1 corresponds to the positive class

        # Get Lung Lime interpretation
        lung_explainer, lung_lime_explanation = lung_lime_interpretation(lung_features)

        # Convert Lung Lime explanation to HTML format
        lung_lime_html = lung_lime_to_html(lung_lime_explanation)

        # Save Lung Lime explanation to HTML file
        lung_html_filename = 'static/lung_lime_visualization.html'
        with open(lung_html_filename, 'w', encoding='utf-8') as file:
            file.write(lung_lime_html)

        # Display the result in a popup
        lung_result = "Lung Disease" if lung_probability_positive_class >= 0.5 else "No Lung Disease"

        return render_template('lung_result.html', lung_result=lung_result, lung_lime_visualization_path=lung_html_filename)

    return render_template('lung_index.html')

@app.route('/predict_diabetes', methods=['POST','GET'])
def predict_diabetes():
    # Get input values from the form for diabetes prediction
    if request.method == 'POST':
        diabetes_features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree_function']),
            float(request.form['age'])
        ]

        if not hasattr(diabetes_model, 'predict_proba') or not callable(getattr(diabetes_model, 'predict_proba', None)):
            raise AttributeError("The loaded diabetes model does not have a 'predict_proba' method.")

        # Make a diabetes prediction
        diabetes_raw_prediction = diabetes_model.predict_proba([diabetes_features])[0]
        diabetes_probability_positive_class = diabetes_raw_prediction[1]  # Assuming 1 corresponds to the positive class

        # Get Diabetes Lime interpretation
        diabetes_explainer, diabetes_lime_explanation = diabetes_lime_interpretation(diabetes_features)

        # Convert Diabetes Lime explanation to HTML format
        diabetes_lime_html = diabetes_lime_to_html(diabetes_lime_explanation)

        # Save Diabetes Lime explanation to HTML file
        diabetes_html_filename = 'static/diabetes_lime_visualization.html'
        with open(diabetes_html_filename, 'w', encoding='utf-8') as file:
            file.write(diabetes_lime_html)

        # Display the result in a popup
        diabetes_result = "Diabetes" if diabetes_probability_positive_class >= 0.5 else "No Diabetes"

        return render_template('diabetes_result.html', diabetes_result=diabetes_result, diabetes_lime_visualization_path=diabetes_html_filename)

@app.route('/diabetes', methods=['POST','GET'])
def diabetes_form():
    return render_template('diabetes_index.html')

generated_history = []

# Route for both GET and POST requests to generate responses
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Handle POST request when form is submitted
        user_input = request.form['user_input']
        sentence = prompt_input.format_map({'input': user_input})
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        
        start_time = time.time()

        with torch.no_grad():
            beam_output = model.generate(**inputs,
                                         min_new_tokens=1, 
                                         max_length=100,
                                         num_beams=1,
                                         repetition_penalty=0.7,
                                         early_stopping=True,
                                         eos_token_id=198 
                                        )

        end_time = time.time()

        output_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
        time_taken = end_time - start_time

        # Add current output and time to history
        generated_history.append((output_text, time_taken))

        return render_template('generate.html', user_input=user_input, output_text=output_text, time_taken=time_taken, generated_history=generated_history)
    
    # Handle GET request to display initial form
    return render_template('generate.html')




from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import matplotlib.pyplot as plt


# Load the saved Keras model for pneumonia prediction
model_path = 'pneumonia.h5'
pneumonia_model = tf.keras.models.load_model(model_path)

# Function to preprocess the image for prediction
def preprocess_image(img_path, img_size=(150, 150)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size used during training
    return img_array

# Route to upload image and display prediction
@app.route('/grad', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Preprocess the image
            img = preprocess_image(file_path)

            # Perform prediction
            predictions = pneumonia_model.predict(img)

            # Since your model is binary classification (sigmoid activation in last layer),
            # predictions will be probabilities in range [0, 1]. Threshold at 0.5 for binary decision.
            prediction = predictions[0][0]
            class_label = 'Pneumonia' if prediction > 0.5 else 'Normal'

            # Display the image and prediction
            plt.figure(figsize=(6, 6))
            plt.imshow(image.load_img(file_path))
            plt.title(f"Predicted: {class_label} ({prediction:.2f})")
            plt.axis('off')

            # Save the plot to a temporary file
            plot_file = 'prediction_plot.png'
            plt.savefig(plot_file)
            plt.close()

            # Encode the image plot to base64 for HTML display
            with open(plot_file, 'rb') as f:
                plot_base64 = base64.b64encode(f.read()).decode('utf-8')

            # Render results in index.html
            return render_template('grad.html', filename=filename, class_label=class_label, plot_base64=plot_base64)

    return render_template('grad.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)




