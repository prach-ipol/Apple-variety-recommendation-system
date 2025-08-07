from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from pymongo import MongoClient
from bson import ObjectId
import json
from datetime import datetime
from twilio.rest import Client

app = Flask(__name__)

# Debug print
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['farming']
    # Drop existing collection to ensure clean data
    db.drop_collection('farming_data')
    collection = db['farming_data']
    # Create a new collection for user inputs
    user_inputs = db['user_inputs']
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None
    db = None
    collection = None
    user_inputs = None

# Twilio setup (replace with your credentials)
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_FROM_NUMBER = '+1234567890'
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# MongoDB collection for notification logs
msg_collection = db['msg']

# Example notification function

def send_sms_and_log(contact, message):
    log = {
        'contact': contact,
        'message': message,
        'timestamp': datetime.now(),
        'status': 'pending'
    }
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=f"+91{contact}"
        )
        log['status'] = 'sent'
    except Exception as e:
        log['status'] = f'error: {str(e)}'
    msg_collection.insert_one(log)
    return log['status']

def convert_rain_level(rain_level):
    # Convert rain level string to numerical value
    rain_level = str(rain_level).strip().lower()
    if rain_level == 'very high':
        return 100
    elif rain_level == 'high':
        return 75
    elif rain_level == 'medium':
        return 50
    elif rain_level == 'low':
        return 25
    else:
        return 50  # default value

def convert_soil_type(soil_type):
    # Convert soil type string to categorical value
    soil_type = str(soil_type).strip().lower()
    if soil_type == 'loamy soil':
        return 'loamy soil'
    elif soil_type == 'black soil':
        return 'black soil'
    else:
        return 'loamy soil'  # default to loamy soil

def convert_soil_type_to_string(soil_type_value):
    # Convert soil type value back to string for display (already a string)
    soil_type_value = str(soil_type_value).strip()
    if soil_type_value.lower() == 'loamy soil':
        return 'Loamy soil'
    elif soil_type_value.lower() == 'black soil':
        return 'Black Soil'
    else:
        return 'Loamy soil'  # default

# Load data into MongoDB if not already present
def load_data_to_mongodb():
    try:
        print("Loading data into MongoDB...")
        # Read CSV file
        df = pd.read_csv('Crop_recommendation (2).csv')
        # Clean column names by removing extra spaces
        df.columns = df.columns.str.strip()
        print("Original columns:", df.columns.tolist())
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'N': 'nitrogen',
            'P': 'phosphorus', 
            'K': 'potassium',
            'rainfall': 'rain_level'
        })
        
        # Convert rain level to numerical values (if needed)
        # The new dataset already has numerical rainfall values
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert records into MongoDB
        result = collection.insert_many(records)
        print(f"Successfully loaded {len(result.inserted_ids)} records into MongoDB")
        
        # Verify data was loaded

        
        count = collection.count_documents({})
        print(f"Total documents in collection: {count}")
        
        # Print first document to verify structure
        first_doc = collection.find_one()
        print("First document structure:", first_doc)
        
    except Exception as e:
        print(f"Error loading data into MongoDB: {e}")
        raise e

# Load data into MongoDB on startup
load_data_to_mongodb()

def prepare_data():
    try:
        print("Preparing data...")
        # Get all records from MongoDB
        records = list(collection.find({}, {'_id': 0}))
        if not records:
            raise Exception("No records found in MongoDB")
            
        df = pd.DataFrame(records)
        print("MongoDB columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        
        # Create a matrix of numerical features for similarity calculation
        # Use the new crop dataset features
        numerical_features = ['temperature', 'humidity', 'rain_level', 'ph', 'nitrogen', 'phosphorus', 'potassium']
        feature_matrix = df[numerical_features].values
        
        # Normalize the features
        feature_matrix = (feature_matrix - feature_matrix.mean()) / feature_matrix.std()
        
        print("Data preparation successful")
        return feature_matrix, df
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise e

def save_user_input(input_data):
    try:
        # Add timestamp to input data
        input_data['timestamp'] = datetime.now()
        # Convert numeric values to float
        for key in ['temperature', 'humidity', 'rain_level', 'ph', 'nitrogen', 'phosphorus', 'potassium']:
            input_data[key] = float(input_data[key])
        
        # Save to MongoDB
        result = user_inputs.insert_one(input_data)
        print(f"Saved user input with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"Error saving user input: {e}")
        raise e

def get_recommendations(input_data, n_recommendations=3):
    try:
        print("Getting recommendations based on input data...")
        print("Input data:", input_data)
        feature_matrix, df = prepare_data()
        
        # Create input feature vector for crop recommendation
        input_features = np.array([
            float(input_data['temperature']),
            float(input_data['humidity']),
            float(input_data['rain_level']),
            float(input_data['ph']),
            float(input_data.get('nitrogen', 50)),  # Default values for missing features
            float(input_data.get('phosphorus', 50)),
            float(input_data.get('potassium', 50))
        ]).reshape(1, -1)
        
        # Normalize input features
        input_features = (input_features - feature_matrix.mean()) / feature_matrix.std()
        
        # Calculate similarity between input and all entries
        similarity_matrix = cosine_similarity(input_features, feature_matrix)
        
        # Get similar entries
        similar_indices = similarity_matrix[0].argsort()[::-1][:n_recommendations]
        
        recommendations = []
        for idx in similar_indices:
            similarity_score = similarity_matrix[0][idx]
            row = df.iloc[idx]
            recommendations.append({
                'crop': row['label'],
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'rain_level': row['rain_level'],
                'ph': row['ph'],
                'nitrogen': row['nitrogen'],
                'phosphorus': row['phosphorus'],
                'potassium': row['potassium'],
                'similarity': float(similarity_score)
            })
        
        print(f"Found {len(recommendations)} recommendations")
        return recommendations
    except Exception as e:
        print(f"Error in recommendations: {e}")
        return [{'error': str(e)}]

@app.route('/')
def home():
    try:
        print("Loading home page...")
        # Get unique crops from MongoDB
        crops = sorted(collection.distinct('label'))
        
        # Get unique soil types (if any)
        soil_types = ['Loamy soil', 'Black soil']  # Default soil types
        
        # Get some statistics about the dataset
        avg_temp = collection.aggregate([
            {'$group': {'_id': None, 'avg': {'$avg': '$temperature'}}}
        ]).try_next()
        avg_temp_value = round(avg_temp['avg'], 1) if avg_temp else 0

        avg_humidity = collection.aggregate([
            {'$group': {'_id': None, 'avg': {'$avg': '$humidity'}}}
        ]).try_next()
        avg_humidity_value = round(avg_humidity['avg'], 1) if avg_humidity else 0

        stats = {
            'total_crops': len(crops),
            'total_entries': collection.count_documents({}),
            'total_user_inputs': user_inputs.count_documents({}),
            'avg_temperature': avg_temp_value,
            'avg_humidity': avg_humidity_value
        }
        
        # Get recent user inputs with their recommendations
        recent_inputs = list(user_inputs.find().sort('timestamp', -1).limit(5))
        for input_data in recent_inputs:
            input_data['recommendations'] = get_recommendations(input_data)
            # No need to convert soil type for crop recommendations
            pass
        
        print(f"Found {len(crops)} crops")
        return render_template('index.html', 
                             crops=crops,
                             soil_types=soil_types,
                             stats=stats,
                             recent_inputs=recent_inputs)
    except Exception as e:
        print(f"Error loading home page: {e}")
        # Provide default values for all expected keys to avoid template errors
        return render_template('index.html', 
                             crops=['Error loading crops'],
                             soil_types=['Error loading soil types'],
                             stats={
                                 'total_crops': 0,
                                 'total_entries': 0,
                                 'total_user_inputs': 0,
                                 'avg_temperature': 0,
                                 'avg_humidity': 0
                             },
                             recent_inputs=[])

@app.route('/get_recommendations', methods=['POST'])
def recommend():
    try:
        # Get input data from form
        input_data = {
            'temperature': request.form['temperature'],
            'humidity': request.form['humidity'],
            'rain_level': request.form['rain_level'],
            'ph': request.form['ph'],
            'nitrogen': request.form['nitrogen'],
            'phosphorus': request.form['phosphorus'],
            'potassium': request.form['potassium']
        }
        
        print(f"Received recommendation request with input data: {input_data}")
        
        # Save user input to MongoDB
        input_id = save_user_input(input_data)
        
        # Get recommendations
        recommendations = get_recommendations(input_data)
        
        # Add input ID to response
        response_data = {
            'input_id': str(input_id),
            'recommendations': recommendations
        }
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in recommend route: {e}")
        return jsonify({
            'error': str(e),
            'recommendations': [{'error': str(e)}]
        })

@app.route('/get_saved_recommendations/<input_id>')
def get_saved_recommendations(input_id):
    try:
        # Get the saved input data
        input_data = user_inputs.find_one({'_id': ObjectId(input_id)})
        if not input_data:
            return jsonify({'error': 'Input data not found'})
        
        # Get recommendations
        recommendations = get_recommendations(input_data)
        
        return jsonify({
            'input_data': input_data,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Error getting saved recommendations: {e}")
        return jsonify({
            'error': str(e),
            'recommendations': [{'error': str(e)}]
        })

@app.route('/send_notifications', methods=['POST'])
def send_notifications():
    # Example: send a test notification to a hardcoded number
    contact = '8010489392'  # Replace with dynamic logic as needed
    message = 'Hello! This is a test notification from your Flask app.'
    status = send_sms_and_log(contact, message)
    return jsonify({'contact': contact, 'status': status})

@app.route('/get_notification_log')
def get_notification_log():
    logs = list(msg_collection.find().sort('timestamp', -1).limit(20))
    for log in logs:
        log['timestamp'] = log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        log['_id'] = str(log['_id'])
    return jsonify(logs)

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
    