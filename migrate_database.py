from pymongo import MongoClient
import json
from datetime import datetime

def migrate_database():
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        
        # Source database (old)
        source_db = client['apple_farming']
        
        # Target database (new)
        target_db = client['farming']
        
        print("Starting database migration from 'apple_farming' to 'farming'...")
        
        # Get all collections from source database
        collections = source_db.list_collection_names()
        print(f"Found collections: {collections}")
        
        for collection_name in collections:
            print(f"Migrating collection: {collection_name}")
            
            # Get all documents from source collection
            source_collection = source_db[collection_name]
            documents = list(source_collection.find({}))
            
            if documents:
                # Create target collection and insert documents
                target_collection = target_db[collection_name]
                result = target_collection.insert_many(documents)
                print(f"Migrated {len(result.inserted_ids)} documents to {collection_name}")
            else:
                print(f"Collection {collection_name} is empty, skipping...")
        
        print("Database migration completed successfully!")
        
        # Verify migration
        print("\nVerification:")
        for collection_name in collections:
            source_count = source_db[collection_name].count_documents({})
            target_count = target_db[collection_name].count_documents({})
            print(f"{collection_name}: {source_count} -> {target_count} documents")
            
    except Exception as e:
        print(f"Error during migration: {e}")

def export_database_to_json():
    """Export the farming database to JSON files for GitHub"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['farming']
        
        collections = db.list_collection_names()
        
        for collection_name in collections:
            collection = db[collection_name]
            documents = list(collection.find({}))
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                # Convert datetime objects to string
                for key, value in doc.items():
                    if isinstance(value, datetime):
                        doc[key] = value.isoformat()
            
            # Save to JSON file
            filename = f"{collection_name}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            print(f"Exported {len(documents)} documents to {filename}")
            
    except Exception as e:
        print(f"Error exporting database: {e}")

if __name__ == "__main__":
    # First migrate the database
    migrate_database()
    
    # Then export to JSON files
    print("\nExporting database to JSON files...")
    export_database_to_json()
    
    print("\nDatabase migration and export completed!") 