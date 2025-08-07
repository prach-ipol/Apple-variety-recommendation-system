from pymongo import MongoClient

def clean_duplicate_data():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['farming']
        
        # Clean farming_data collection - remove duplicates
        collection = db['farming_data']
        
        # Count before cleaning
        before_count = collection.count_documents({})
        print(f"Before cleaning: {before_count} documents")
        
        # Remove all documents and reload from CSV
        collection.delete_many({})
        
        # Reload data from CSV (this will be done by app.py when it starts)
        print("Cleaned farming_data collection. Data will be reloaded when app.py starts.")
        
        # Count after cleaning
        after_count = collection.count_documents({})
        print(f"After cleaning: {after_count} documents")
        
    except Exception as e:
        print(f"Error cleaning database: {e}")

if __name__ == "__main__":
    clean_duplicate_data() 