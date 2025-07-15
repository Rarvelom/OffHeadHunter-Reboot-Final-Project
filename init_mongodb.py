from mongodb_schema import (
    users, job_sources, job_offers, cv_uploads, cv_rewrites,
    applications, notifications, activity_logs, job_matches,
    chat_history, chat_memory   
)
from pymongo import IndexModel, ASCENDING, DESCENDING

def create_indexes():
    # Users indexes
    users.create_indexes([
        IndexModel([('email', ASCENDING)], unique=True),
        IndexModel([('preferences.desired_position', ASCENDING)]),
        IndexModel([('preferences.locations.country', ASCENDING)])
    ])

    # Job Offers indexes
    job_offers.create_indexes([
        IndexModel([('source_id', ASCENDING)]),
        IndexModel([('title', ASCENDING)]),
        IndexModel([('company', ASCENDING)]),
        IndexModel([('tags', ASCENDING)]),
        IndexModel([('is_active', ASCENDING)]),
        IndexModel([('locations.country', ASCENDING)]),
        IndexModel([('locations.work_mode', ASCENDING)])
    ])

    # Job Matches indexes
    job_matches.create_indexes([
        IndexModel([('user_id', ASCENDING)]),
        IndexModel([('job_offer_id', ASCENDING)]),
        IndexModel([('score', DESCENDING)]),
        IndexModel([('matched_at', DESCENDING)]),
        IndexModel([('is_recommended', ASCENDING)])
    ])

    # Chat History indexes
    chat_history.create_indexes([
        IndexModel([('user_id', ASCENDING)]),
        IndexModel([('conversation_id', ASCENDING)]),
        IndexModel([('timestamp', DESCENDING)]),
        IndexModel([('message_type', ASCENDING)])
    ])

    # Chat Memory indexes
    chat_memory.create_indexes([
        IndexModel([('key', ASCENDING)], unique=True),
        IndexModel([('key', ASCENDING), ('chatHistory.role', ASCENDING)]),
        IndexModel([('chatHistory.role', ASCENDING), ('chatHistory.content', ASCENDING)]),
        IndexModel([('chatHistory.role', ASCENDING)])
    ])

    # CV Uploads indexes
    cv_uploads.create_indexes([
        IndexModel([('user_id', ASCENDING)]),
        IndexModel([('vectorized', ASCENDING)]),
        IndexModel([('embedding_model', ASCENDING)])
    ])

    # CV Rewrites indexes
    cv_rewrites.create_indexes([
        IndexModel([('user_id', ASCENDING)]),
        IndexModel([('job_offer_id', ASCENDING)]),
        IndexModel([('original_cv_id', ASCENDING)])
    ])

    # Applications indexes
    applications.create_indexes([
        IndexModel([('user_id', ASCENDING)]),
        IndexModel([('job_offer_id', ASCENDING)]),
        IndexModel([('status', ASCENDING)]),
        IndexModel([('updated_at', DESCENDING)])
    ])

    # Notifications indexes
    notifications.create_indexes([
        IndexModel([('user_id', ASCENDING)]),
        IndexModel([('read', ASCENDING)]),
        IndexModel([('created_at', DESCENDING)])
    ])

    # Activity Logs indexes
    activity_logs.create_indexes([
        IndexModel([('user_id', ASCENDING)]),
        IndexModel([('action', ASCENDING)]),
        IndexModel([('timestamp', DESCENDING)])
    ])

def main():
    print("Creating collections and indexes...")
    create_indexes()
    print("Collections and indexes created successfully!")

if __name__ == "__main__":
    main()
