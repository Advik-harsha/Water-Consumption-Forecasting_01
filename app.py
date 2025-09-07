import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
import os

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from form
            population = int(request.form['population'])
            industries = int(request.form['industries'])
            temperature = float(request.form['temperature'])
            irrigation = int(request.form['irrigation'])
            rainfall = int(request.form['rainfall'])
            
            # Create input data dictionary with feature names that match the training data
            # This is the key fix - mapping form field names to the feature names used during model training
            input_data = {
                'Population': population,
                'Number_of_Industries': industries,
                'Average_Temperature': temperature,      # Changed to match CSV column name
                'Irrigation_Area': irrigation,           # Changed to match CSV column name
                'Rainfall': rainfall                     # This matches the CSV column name
            }
            
            # For display purposes in the template
            display_data = {
                'Population': population,
                'Number of Industries': industries,
                'Average Temperature (°C)': temperature,
                'Irrigation Area (hectares)': irrigation,
                'Annual Rainfall (mm)': rainfall
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction = round(prediction, 2)
            
            # Generate comparison chart
            generate_comparison_chart(input_data)
            
            return render_template('predict.html', prediction=prediction, input_data=display_data)
        
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

def generate_comparison_chart(input_data):
    # Load historical data
    df = pd.read_csv('water_data.csv')
    
    # Calculate averages
    avg_population = df['Population'].mean()
    avg_industries = df['Number_of_Industries'].mean()
    avg_temperature = df['Average_Temperature'].mean()
    avg_irrigation = df['Irrigation_Area'].mean()
    avg_rainfall = df['Rainfall'].mean()
    
    # Create comparison data
    categories = ['Population', 'Industries', 'Temperature', 'Irrigation', 'Rainfall']
    historical = [avg_population/1000000, avg_industries/100, avg_temperature, avg_irrigation/1000, avg_rainfall/100]
    current = [input_data['Population']/1000000, input_data['Number_of_Industries']/100, 
              input_data['Average_Temperature'], input_data['Irrigation_Area']/1000, 
              input_data['Rainfall']/100]
    
    # Create comparison chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, historical, width, label='Historical Average')
    plt.bar(x + width/2, current, width, label='Current Input')
    
    plt.xlabel('Factors')
    plt.ylabel('Normalized Values')
    plt.title('Comparison with Historical Average')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig('static/images/comparison.png')
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)