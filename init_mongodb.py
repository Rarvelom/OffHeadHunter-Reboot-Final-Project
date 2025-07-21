from mongodb_schema import (
    User, JobSource, JobOffer, CVUpload, CVRewrite,
    Application, Notification, ActivityLog, JobMatch,
    ChatHistory, ChatMemory, AgentTestQuery
)
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT

def create_collections_with_validation(db):
    """Create collections with validation schemas."""
    # Get collection configurations from schema classes
    collections_config = {
        'users': User.collection_config,
        'job_sources': JobSource.collection_config,
        'job_offers': JobOffer.collection_config,
        'cv_uploads': CVUpload.collection_config,
        'cv_rewrites': CVRewrite.collection_config,
        'job_matches': JobMatch.collection_config,
        'applications': Application.collection_config,
        'notifications': Notification.collection_config,
        'activity_logs': ActivityLog.collection_config,
        'chat_history': ChatHistory.collection_config,
        'chat_memory': ChatMemory.collection_config,
        'agent_test_queries': AgentTestQuery.collection_config
    }

    for coll_name, config in collections_config.items():
        # Create collection with validation
        db.create_collection(
            config['collection_name'],
            validator={
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': [k for k, v in config['schema'].items() if v.get('required', False)],
                    'properties': {
                        k: v for k, v in config['schema'].items()
                        if not k.startswith('_')
                    }
                }
            },
            validationLevel=config.get('validation_level', 'strict'),
            validationAction=config.get('validation_action', 'error')
        )

def create_indexes(db):
    """Create indexes for all collections."""
    # Users indexes
    db.users.create_indexes([
        IndexModel([('email', ASCENDING)], unique=True, background=True),
        IndexModel([('preferences.desired_position', ASCENDING)], background=True),
        IndexModel([('preferences.locations.country', ASCENDING)], background=True),
        IndexModel([('preferences.salary_range.min', ASCENDING), 
                   ('preferences.salary_range.max', ASCENDING)], background=True)
    ])

    # Job Sources indexes
    db.job_sources.create_indexes([
        IndexModel([('name', ASCENDING)], unique=True, background=True),
        IndexModel([('is_active', ASCENDING)], background=True)
    ])

    # Job Offers indexes
    db.job_offers.create_indexes([
        IndexModel([('source_id', ASCENDING)], background=True),
        IndexModel([('title', 'text'), ('description', 'text'), ('company', 'text')], 
                  weights={'title': 10, 'company': 5, 'description': 1}),
        IndexModel([('company', ASCENDING)], background=True),
        IndexModel([('tags', ASCENDING)], background=True),
        IndexModel([('is_active', ASCENDING)], background=True),
        IndexModel([('locations.country', ASCENDING)], background=True),
        IndexModel([('locations.work_mode', ASCENDING)], background=True),
        IndexModel([('salary.min', ASCENDING), ('salary.max', ASCENDING)], background=True),
        IndexModel([('publication_date', DESCENDING)], background=True)
    ])

    # CV Uploads indexes
    db.cv_uploads.create_indexes([
        IndexModel([('user_id', ASCENDING)], background=True),
        IndexModel([('is_processed', ASCENDING)], background=True),
        IndexModel([('embedding_model', ASCENDING)], background=True),
        IndexModel([('upload_date', DESCENDING)], background=True)
    ])

    # CV Rewrites indexes
    db.cv_rewrites.create_indexes([
        IndexModel([('user_id', ASCENDING)], background=True),
        IndexModel([('job_offer_id', ASCENDING)], background=True),
        IndexModel([('original_cv_id', ASCENDING)], background=True),
        IndexModel([('created_at', DESCENDING)], background=True)
    ])

    # Job Matches indexes
    db.job_matches.create_indexes([
        IndexModel([('user_id', ASCENDING), ('job_offer_id', ASCENDING)], unique=True, background=True),
        IndexModel([('user_id', ASCENDING), ('is_recommended', ASCENDING), ('score', DESCENDING)], background=True),
        IndexModel([('score', DESCENDING)], background=True),
        IndexModel([('matched_at', DESCENDING)], background=True)
    ])

    # Applications indexes
    db.applications.create_indexes([
        IndexModel([('user_id', ASCENDING), ('job_offer_id', ASCENDING)], unique=True, background=True),
        IndexModel([('status', ASCENDING), ('updated_at', DESCENDING)], background=True),
        IndexModel([('applied_at', DESCENDING)], background=True)
    ])

    # Notifications indexes
    db.notifications.create_indexes([
        IndexModel([('user_id', ASCENDING), ('read', ASCENDING), ('created_at', DESCENDING)], background=True),
        IndexModel([('related_entity.entity_type', ASCENDING), 
                   ('related_entity.entity_id', ASCENDING)], sparse=True, background=True),
        IndexModel([('expires_at', ASCENDING)], expireAfterSeconds=0, background=True)
    ])

    # Activity Logs indexes
    db.activity_logs.create_indexes([
        IndexModel([('user_id', ASCENDING)], background=True),
        IndexModel([('action', ASCENDING)], background=True),
        IndexModel([('entity_type', ASCENDING), ('entity_id', ASCENDING)], background=True),
        IndexModel([('timestamp', DESCENDING)], background=True),
        IndexModel([('timestamp', ASCENDING)], expireAfterSeconds=60*60*24*30*6, background=True)  # 6 months TTL
    ])

    # Chat History indexes
    db.chat_history.create_indexes([
        IndexModel([('user_id', ASCENDING)], background=True),
        IndexModel([('session_id', ASCENDING)], background=True),
        IndexModel([('timestamp', DESCENDING)], background=True),
        IndexModel([('message_type', ASCENDING)], background=True)
    ])

    # Chat Memory indexes
    db.chat_memory.create_indexes([
        IndexModel([('key', ASCENDING)], unique=True, background=True),
        IndexModel([('key', ASCENDING), ('chatHistory.role', ASCENDING)], background=True),
        IndexModel([('last_accessed', DESCENDING)], background=True)
    ])

    # Agent Test Queries indexes
    db.agent_test_queries.create_indexes([
        IndexModel([('session_id', ASCENDING)], background=True),
        IndexModel([('timestamp', DESCENDING)], background=True),
        IndexModel([('job_title', TEXT)], background=True)
    ])

def get_database():
    """Create and return a MongoDB database connection."""
    from pymongo import MongoClient
    import os
    
    # Get MongoDB connection string from environment or use default
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGODB_DB_NAME', 'offheadhunter')
    
    client = MongoClient(mongo_uri)
    return client[db_name]

def main():
    print("Initializing MongoDB database...")
    
    # Get database connection
    db = get_database()
    
    try:
        # Create collections with validation
        print("Creating collections with validation...")
        create_collections_with_validation(db)
        
        # Create indexes
        print("Creating indexes...")
        create_indexes(db)
        
        print("\nDatabase initialization completed successfully!")
        print(f"\nCollections created in database '{db.name}':")
        for coll_name in db.list_collection_names():
            print(f"- {coll_name}")
            
    except Exception as e:
        print(f"\nError during database initialization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
