from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from pymongo import MongoClient
from bson import ObjectId
import json
from datetime import datetime

app = Flask(__name__)

# Debug print
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['apple_farming']
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

# Load data into MongoDB if not already present
def load_data_to_mongodb():
    try:
        print("Loading data into MongoDB...")
        # Read CSV file
        df = pd.read_csv('dataset1.csv')
        # Clean column names by removing extra spaces
        df.columns = df.columns.str.strip()
        print("Original columns:", df.columns.tolist())
        
        # Convert rain level to numerical values
        df['rain level'] = df['rain level'].apply(convert_rain_level)
        
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
        numerical_features = ['Age', 'temperature', 'humidity', 'rain level', 'ph', 'apple yeild in first season']
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
        for key in ['age', 'temperature', 'humidity', 'rain_level', 'ph', 'yield']:
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
        
        # Create input feature vector
        input_features = np.array([
            float(input_data['age']),
            float(input_data['temperature']),
            float(input_data['humidity']),
            float(input_data['rain_level']),
            float(input_data['ph']),
            float(input_data['yield'])
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
            print("Row data:", row.to_dict())
            recommendations.append({
                'state': row['State'],
                'region': row['Region'],
                'plant_variety': row['Plant variety'],
                'apple_variety': row['Apple variety'],
                'age': row['Age'],
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'rain_level': row['rain level'],
                'soil_type': row['soil type'],
                'ph': row['ph'],
                'yield': row['apple yeild in first season'],
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
        # Get unique states and regions from MongoDB
        states = sorted(collection.distinct('State'))
        regions = sorted(collection.distinct('Region'))
        
        # Get some statistics about the dataset
        stats = {
            'total_states': len(states),
            'total_regions': len(regions),
            'total_entries': collection.count_documents({}),
            'total_user_inputs': user_inputs.count_documents({}),
            'avg_yield': collection.aggregate([
                {'$group': {'_id': None, 'avg': {'$avg': '$apple yeild in first season'}}}
            ]).next()['avg'],
            'avg_temperature': collection.aggregate([
                {'$group': {'_id': None, 'avg': {'$avg': '$temperature'}}}
            ]).next()['avg'],
            'avg_humidity': collection.aggregate([
                {'$group': {'_id': None, 'avg': {'$avg': '$humidity'}}}
            ]).next()['avg']
        }
        
        # Get recent user inputs with their recommendations
        recent_inputs = list(user_inputs.find().sort('timestamp', -1).limit(5))
        for input_data in recent_inputs:
            input_data['recommendations'] = get_recommendations(input_data)
        
        print(f"Found {len(states)} states and {len(regions)} regions")
        return render_template('index.html', 
                             states=states, 
                             regions=regions,
                             stats=stats,
                             recent_inputs=recent_inputs)
    except Exception as e:
        print(f"Error loading home page: {e}")
        return render_template('index.html', 
                             states=['Error loading states'], 
                             regions=['Error loading regions'],
                             stats={'error': str(e)},
                             recent_inputs=[])

@app.route('/get_recommendations', methods=['POST'])
def recommend():
    try:
        # Get input data from form
        input_data = {
            'age': request.form['age'],
            'temperature': request.form['temperature'],
            'humidity': request.form['humidity'],
            'rain_level': request.form['rain_level'],
            'ph': request.form['ph'],
            'yield': request.form['yield']
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

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000) 