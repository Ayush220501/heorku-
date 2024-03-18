# Import necessary libraries

from flask import Flask, render_template, request
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load the trained ML model
model = joblib.load('fish_weight_prediction_model.pkl')

# Define route to render input form
@app.route('/')
def input_form():
    return render_template('fish.html')

# Define route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        species = request.form['species']
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])

        # Make predictions using the loaded ML model
        input_data = [[length1, length2, length3, height, width]]
        prediction = model.predict(input_data)

        # Render the prediction result template and pass the prediction
        return render_template('fish.html', species=species, prediction=prediction)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
