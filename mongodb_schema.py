from datetime import datetime
from typing import List, Dict, Optional, Union
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Union

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI')
client = MongoClient(MONGODB_URI)
db = client['offheadhunter_db']

class User:
    """
    Schema for user accounts with MongoDB validation rules.
    All fields except 'last_login' are required for new users.
    """
    schema = {
        'email': {
            'type': 'string',
            'required': True,
            'description': 'User email address (must be unique)',
            'regex': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'maxlength': 255
        },
        'password_hash': {
            'type': 'string',
            'required': True,
            'description': 'Hashed password (using bcrypt or similar)',
            'minlength': 60,  # bcrypt hash length
            'maxlength': 100
        },
        'name': {
            'type': 'string',
            'required': True,
            'description': 'Full name of the user',
            'minlength': 2,
            'maxlength': 100
        },
        'preferences': {
            'type': 'object',
            'required': True,
            'description': 'User job search preferences',
            'properties': {
                'desired_position': {
                    'type': 'string',
                    'required': True,
                    'description': 'Job title or position the user is seeking',
                    'minlength': 2,
                    'maxlength': 100
                },
                'salary_range': {
                    'type': 'object',
                    'required': True,
                    'properties': {
                        'min': {
                            'type': 'number',
                            'required': True,
                            'minimum': 0,
                            'description': 'Minimum expected salary'
                        },
                        'max': {
                            'type': 'number',
                            'required': True,
                            'minimum': 0,
                            'description': 'Maximum expected salary'
                        },
                        'currency': {
                            'type': 'string',
                            'required': True,
                            'enum': ['EUR', 'USD', 'GBP', 'JPY', 'CAD', 'AUD'],
                            'description': 'Currency code (ISO 4217)'
                        }
                    }
                },
                'locations': {
                    'type': 'array',
                    'required': True,
                    'minItems': 1,
                    'description': 'Preferred work locations',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'country': {
                                'type': 'string',
                                'required': True,
                                'minlength': 2,
                                'maxlength': 100,
                                'description': 'Country name'
                            },
                            'region': {
                                'type': 'string',
                                'required': False,
                                'maxlength': 100,
                                'description': 'State/Region/Province'
                            },
                            'city': {
                                'type': 'string',
                                'required': False,
                                'maxlength': 100,
                                'description': 'City name'
                            },
                            'work_mode': {
                                'type': 'string',
                                'required': True,
                                'enum': ['onsite', 'remote', 'hybrid'],
                                'description': 'Preferred work mode'
                            }
                        }
                    }
                },
                'job_sources': {
                    'type': 'array',
                    'required': True,
                    'description': 'List of preferred job board IDs',
                    'items': {
                        'type': 'string',
                        'minlength': 1,
                        'maxlength': 100
                    }
                }
            }
        },
        'created_at': {
            'type': 'date',
            'required': True,
            'description': 'Timestamp when the user account was created'
        },
        'last_login': {
            'type': 'date',
            'required': False,
            'description': 'Timestamp of the last successful login'
        },
        'status': {
            'type': 'string',
            'required': True,
            'enum': ['active', 'inactive', 'suspended', 'pending_verification'],
            'default': 'pending_verification',
            'description': 'Account status'
        },
        'email_verified': {
            'type': 'boolean',
            'required': True,
            'default': False,
            'description': 'Whether the email has been verified'
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'users',
        'indexes': [
            {'key': [('email', 1)], 'unique': True},
            {'key': [('created_at', -1)]},
            {'key': [('preferences.desired_position', 'text')]},
            {'key': [('preferences.locations.country', 1)]},
            {'key': [('status', 1)]}
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class JobSource:
    """
    Schema for job sources/boards with scraping configuration.
    All fields are required for new job sources.
    """
    schema = {
        'name': {
            'type': 'string',
            'required': True,
            'description': 'Name of the job board/source',
            'minlength': 2,
            'maxlength': 100
        },
        'base_url': {
            'type': 'string',
            'required': True,
            'description': 'Base URL of the job board',
            'format': 'uri',
            'maxlength': 1000
        },
        'enabled': {
            'type': 'boolean',
            'required': True,
            'default': True,
            'description': 'Whether this job source is currently active'
        },
        'scraping_config': {
            'type': 'object',
            'required': True,
            'description': 'Configuration for web scraping this job source',
            'properties': {
                'filters_supported': {
                    'type': 'array',
                    'required': True,
                    'description': 'List of supported filter types',
                    'items': {
                        'type': 'string',
                        'enum': ['location', 'salary', 'job_type', 'experience_level', 'remote']
                    }
                },
                'selectors': {
                    'type': 'object',
                    'required': True,
                    'description': 'CSS/XPATH selectors for scraping',
                    'properties': {
                        'job_list': {'type': 'string', 'required': True},
                        'job_title': {'type': 'string', 'required': True},
                        'company': {'type': 'string', 'required': True},
                        'location': {'type': 'string', 'required': True},
                        'description': {'type': 'string', 'required': True},
                        'posted_date': {'type': 'string', 'required': False},
                        'salary': {'type': 'string', 'required': False}
                    }
                },
                'pagination': {
                    'type': 'object',
                    'required': False,
                    'description': 'Pagination configuration',
                    'properties': {
                        'type': {
                            'type': 'string',
                            'enum': ['query_param', 'path_param', 'infinite_scroll'],
                            'default': 'query_param'
                        },
                        'param_name': {'type': 'string'},
                        'start_page': {'type': 'number', 'default': 1}
                    }
                }
            }
        },
        'created_at': {
            'type': 'date',
            'required': True,
            'description': 'When this job source was added to the system'
        },
        'last_scraped': {
            'type': 'date',
            'required': False,
            'description': 'When this job source was last successfully scraped'
        },
        'scraping_interval': {
            'type': 'number',
            'required': True,
            'default': 3600,  # 1 hour in seconds
            'description': 'How often to scrape this source (in seconds)'
        },
        'status': {
            'type': 'string',
            'required': True,
            'enum': ['active', 'inactive', 'error', 'maintenance'],
            'default': 'active',
            'description': 'Current status of the job source'
        },
        'error_count': {
            'type': 'number',
            'required': True,
            'default': 0,
            'minimum': 0,
            'description': 'Number of consecutive scraping errors'
        },
        'last_error': {
            'type': 'string',
            'required': False,
            'description': 'Last error message encountered'
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'job_sources',
        'indexes': [
            {'key': [('name', 1)], 'unique': True},
            {'key': [('base_url', 1)], 'unique': True},
            {'key': [('enabled', 1)]},
            {'key': [('last_scraped', 1)]},
            {'key': [('status', 1)]}
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class JobOffer:
    """
    Schema for job offers with detailed validation rules.
    All fields except 'external_id', 'posted_at', 'expires_at', and 'embedding' are required.
    """
    schema = {
        'external_id': {
            'type': 'string',
            'required': False,
            'description': 'ID from the external job board',
            'maxlength': 255
        },
        'source_id': {
            'type': 'string',
            'required': True,
            'description': 'Reference to the job source',
            'minlength': 24,
            'maxlength': 24  # MongoDB ObjectId length
        },
        'title': {
            'type': 'string',
            'required': True,
            'description': 'Job title/position',
            'minlength': 2,
            'maxlength': 255
        },
        'company': {
            'type': 'string',
            'required': True,
            'description': 'Company name',
            'minlength': 2,
            'maxlength': 255
        },
        'locations': {
            'type': 'array',
            'required': True,
            'minItems': 1,
            'description': 'Job locations',
            'items': {
                'type': 'object',
                'properties': {
                    'country': {
                        'type': 'string',
                        'required': True,
                        'minlength': 2,
                        'maxlength': 100,
                        'description': 'Country name'
                    },
                    'region': {
                        'type': 'string',
                        'required': False,
                        'maxlength': 100,
                        'description': 'State/Region/Province'
                    },
                    'city': {
                        'type': 'string',
                        'required': False,
                        'maxlength': 100,
                        'description': 'City name'
                    },
                    'work_mode': {
                        'type': 'string',
                        'required': True,
                        'enum': ['onsite', 'remote', 'hybrid'],
                        'description': 'Work mode for this location'
                    },
                    'address': {
                        'type': 'string',
                        'required': False,
                        'maxlength': 500,
                        'description': 'Full work address'
                    },
                    'coordinates': {
                        'type': 'object',
                        'required': False,
                        'description': 'Geographic coordinates',
                        'properties': {
                            'type': {'type': 'string', 'enum': ['Point'], 'default': 'Point'},
                            'coordinates': {
                                'type': 'array',
                                'minItems': 2,
                                'maxItems': 2,
                                'items': {'type': 'number'}
                            }
                        }
                    }
                }
            }
        },
        'description': {
            'type': 'string',
            'required': True,
            'minlength': 10,
            'description': 'Full job description in HTML or plain text'
        },
        'url': {
            'type': 'string',
            'required': True,
            'description': 'URL to the original job posting',
            'format': 'uri',
            'maxlength': 2000
        },
        'posted_at': {
            'type': 'date',
            'required': False,
            'description': 'When the job was originally posted'
        },
        'scraped_at': {
            'type': 'date',
            'required': True,
            'description': 'When the job was scraped from the source'
        },
        'tags': {
            'type': 'array',
            'required': True,
            'description': 'Job tags/categories',
            'items': {
                'type': 'string',
                'maxlength': 100
            }
        },
        'salary_range': {
            'type': 'object',
            'required': True,
            'description': 'Salary information',
            'properties': {
                'min': {
                    'type': 'number',
                    'required': False,
                    'minimum': 0,
                    'description': 'Minimum salary'
                },
                'max': {
                    'type': 'number',
                    'required': False,
                    'minimum': 0,
                    'description': 'Maximum salary'
                },
                'currency': {
                    'type': 'string',
                    'required': True,
                    'enum': ['EUR', 'USD', 'GBP', 'JPY', 'CAD', 'AUD'],
                    'description': 'Currency code (ISO 4217)'
                },
                'period': {
                    'type': 'string',
                    'required': True,
                    'enum': ['hour', 'day', 'week', 'month', 'year'],
                    'description': 'Payment period'
                },
                'type': {
                    'type': 'string',
                    'required': False,
                    'enum': ['gross', 'net'],
                    'description': 'Whether salary is gross or net'
                }
            }
        },
        'is_active': {
            'type': 'boolean',
            'required': True,
            'default': True,
            'description': 'Whether the job posting is still active'
        },
        'expires_at': {
            'type': 'date',
            'required': False,
            'description': 'When the job posting expires'
        },
        'embedding': {
            'type': 'array',
            'required': False,
            'description': 'Vector embedding of the job description',
            'items': {
                'type': 'number',
                'description': 'Vector dimension value'
            }
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'job_offers',
        'indexes': [
            {'key': [('source_id', 1)]},
            {'key': [('company', 1)]},
            {'key': [('is_active', 1)]},
            {'key': [('expires_at', 1)]},
            {'key': [('scraped_at', -1)]},
            {'key': [('title', 'text'), ('description', 'text'), ('company', 'text')]},
            {
                'key': [('locations.country', 1), ('locations.region', 1), ('locations.city', 1)]
            },
            {
                'key': [('locations.coordinates', '2dsphere')],
                'sparse': True
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class CVUpload:
    """
    Schema for CV uploads with MongoDB validation rules.
    All fields except 'embedding', 'embedding_vector_id', and 'embedding_model' are required.
    """
    schema = {
        'user_id': {
            'type': 'string',
            'required': True,
            'description': 'ID of the user who uploaded the CV',
            'minlength': 1,
            'maxlength': 100
        },
        'filename': {
            'type': 'string',
            'required': True,
            'description': 'Original filename of the uploaded CV',
            'minlength': 1,
            'maxlength': 255
        },
        'file_url': {
            'type': 'string',
            'required': True,
            'description': 'URL or path where the CV file is stored',
            'minlength': 1,
            'maxlength': 1000
        },
        'original_text': {
            'type': 'string',
            'required': True,
            'description': 'Extracted text content from the CV',
            'minlength': 10  # At least some meaningful content
        },
        'version': {
            'type': 'int',
            'required': True,
            'description': 'Version number of the CV (starts at 1)',
            'min': 1
        },
        'vectorized': {
            'type': 'bool',
            'required': True,
            'description': 'Whether the CV has been processed into vector embeddings'
        },
        'embedding': {
            'type': 'array',
            'required': False,
            'description': 'Vector embedding of the CV content',
            'items': {
                'type': 'number',
                'description': 'Vector dimension value'
            }
        },
        'embedding_vector_id_qdrant': {
            'type': 'string',
            'required': True,
            'description': 'Reference ID for the vector in the Qdrant vector database',
            'minlength': 1,
            'maxlength': 100
        },
        'embedding_model': {
            'type': 'string',
            'required': False,
            'description': 'Name/version of the model used to generate the embedding',
            'maxlength': 100
        },
        'uploaded_at': {
            'type': 'date',
            'required': True,
            'description': 'Timestamp when the CV was uploaded'
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional metadata about the CV',
            'properties': {
                'file_size': {'type': 'int', 'minimum': 1},
                'file_type': {'type': 'string', 'enum': ['pdf', 'docx', 'doc', 'txt']},
                'pages': {'type': 'int', 'minimum': 1},
                'language': {'type': 'string', 'maxlength': 10}
            }
        },
        'status': {
            'type': 'string',
            'required': True,
            'enum': ['pending', 'processing', 'processed', 'error'],
            'default': 'pending',
            'description': 'Current processing status of the CV'
        },
        'error': {
            'type': 'string',
            'required': False,
            'description': 'Error message if processing failed'
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'cv_uploads',
        'indexes': [
            # Single field indexes
            {'key': [('user_id', 1)]},
            {'key': [('uploaded_at', -1)]},
            {'key': [('vectorized', 1)]},
            
            # Compound index for common queries
            {
                'key': [
                    ('user_id', 1),
                    ('version', -1)
                ],
                'unique': True
            },
            
            # Text index for search
            {
                'key': [('original_text', 'text')],
                'weights': {
                    'original_text': 10,
                    'filename': 5
                },
                'default_language': 'spanish',
                'name': 'cv_text_search'
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class CVRewrite:
    """
    Schema for CV Rewrite documents.
    Stores different versions of a CV that have been rewritten or adapted for specific job applications.
    """
    schema = {
        '_id': Optional[str],  # MongoDB generated ObjectId
        'user_id': {
            'type': str,
            'required': True,
            'description': 'ID of the user who owns this CV rewrite',
            'minlength': 1,
            'maxlength': 100
        },
        'original_cv_id': {
            'type': str,
            'required': True,
            'description': 'Reference to the original CV document',
            'minlength': 1,
            'maxlength': 100
        },
        'job_offer_id': {
            'type': str,
            'required': True,
            'description': 'Reference to the job offer this CV was adapted for',
            'minlength': 1,
            'maxlength': 100
        },
        'content': {
            'type': str,
            'required': True,
            'description': 'The rewritten CV content',
            'minlength': 10  # Minimum length to ensure meaningful content
        },
        'version': {
            'type': int,
            'required': True,
            'description': 'Version number of this rewrite',
            'minimum': 1
        },
        'generated_by': {
            'type': str,
            'required': True,
            'description': 'Source of this rewrite (e.g., "user", "ai_assistant", "template")',
            'enum': ['user', 'ai_assistant', 'template', 'system']
        },
        'created_at': {
            'type': datetime,
            'required': True,
            'description': 'When this version was created'
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional metadata about this rewrite',
            'properties': {
                'model_used': {'type': str, 'required': False},
                'prompt_used': {'type': str, 'required': False},
                'changes_made': {'type': 'array', 'items': str, 'required': False}
            }
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'cv_rewrites',
        'indexes': [
            {'key': [('user_id', 1)]},
            {'key': [('original_cv_id', 1)]},
            {'key': [('job_offer_id', 1)]},
            {'key': [('created_at', -1)]},
            {
                'key': [
                    ('user_id', 1),
                    ('job_offer_id', 1),
                    ('version', -1)
                ],
                'unique': True
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class JobMatch:
    """
    Schema for Job Match documents.
    Tracks matches between job seekers and job offers with matching scores and feedback.
    """
    schema = {
        '_id': Optional[str],  # MongoDB generated ObjectId
        'user_id': {
            'type': str,
            'required': True,
            'description': 'ID of the user who is being matched',
            'minlength': 1,
            'maxlength': 100
        },
        'job_offer_id': {
            'type': str,
            'required': True,
            'description': 'ID of the job offer being matched',
            'minlength': 1,
            'maxlength': 100
        },
        'score': {
            'type': 'number',
            'required': True,
            'description': 'Matching score (0-100)',
            'minimum': 0,
            'maximum': 100
        },
        'match_algorithm': {
            'type': 'string',
            'required': True,
            'description': 'Algorithm used to calculate the match',
            'enum': ['semantic', 'keyword', 'hybrid', 'manual']
        },
        'is_recommended': {
            'type': 'boolean',
            'required': True,
            'description': 'Whether this match is recommended to the user',
            'default': False
        },
        'user_feedback': {
            'type': 'string',
            'required': False,
            'description': 'Feedback provided by the user about this match',
            'maxlength': 2000
        },
        'matched_at': {
            'type': 'date',
            'required': True,
            'description': 'When this match was calculated'
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional metadata about the match',
            'properties': {
                'skill_match': {'type': 'number', 'minimum': 0, 'maximum': 100},
                'experience_match': {'type': 'number', 'minimum': 0, 'maximum': 100},
                'location_match': {'type': 'number', 'minimum': 0, 'maximum': 100},
                'salary_match': {'type': 'number', 'minimum': 0, 'maximum': 100}
            }
        },
        'status': {
            'type': 'string',
            'required': True,
            'enum': ['new', 'viewed', 'applied', 'rejected', 'archived'],
            'default': 'new',
            'description': 'Current status of the match from user\'s perspective'
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'job_matches',
        'indexes': [
            # Single field indexes
            {'key': [('user_id', 1)]},
            {'key': [('job_offer_id', 1)]},
            {'key': [('score', -1)]},
            {'key': [('is_recommended', 1)]},
            {'key': [('status', 1)]},
            {'key': [('matched_at', -1)]},
            
            # Compound indexes
            {
                'key': [
                    ('user_id', 1),
                    ('job_offer_id', 1)
                ],
                'unique': True
            },
            {
                'key': [
                    ('user_id', 1),
                    ('is_recommended', 1),
                    ('score', -1)
                ]
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class Application:
    """
    Schema for Job Application documents.
    Tracks the entire lifecycle of a job application from submission to final decision.
    """
    schema = {
        '_id': Optional[str],  # MongoDB generated ObjectId
        'user_id': {
            'type': str,
            'required': True,
            'description': 'ID of the user who submitted the application',
            'minlength': 1,
            'maxlength': 100
        },
        'job_offer_id': {
            'type': str,
            'required': True,
            'description': 'ID of the job offer being applied to',
            'minlength': 1,
            'maxlength': 100
        },
        'cv_rewrite_id': {
            'type': str,
            'required': False,
            'description': 'Reference to the specific CV version used for this application',
            'minlength': 1,
            'maxlength': 100
        },
        'status': {
            'type': 'string',
            'required': True,
            'description': 'Current status of the application',
            'enum': [
                'draft', 'submitted', 'under_review', 'shortlisted',
                'interviewing', 'offer_pending', 'hired', 'rejected', 'withdrawn'
            ],
            'default': 'draft'
        },
        'cover_letter': {
            'type': 'string',
            'required': False,
            'description': 'Cover letter content',
            'maxlength': 10000
        },
        'notes': {
            'type': 'string',
            'required': False,
            'description': 'Internal notes about the application',
            'maxlength': 5000
        },
        'interview_date': {
            'type': 'date',
            'required': False,
            'description': 'Scheduled date and time for the interview'
        },
        'stage_history': {
            'type': 'array',
            'required': True,
            'description': 'History of all status changes for this application',
            'items': {
                'type': 'object',
                'required': ['status', 'timestamp'],
                'properties': {
                    'status': {
                        'type': 'string',
                        'description': 'Status at this stage'
                    },
                    'timestamp': {
                        'type': 'date',
                        'description': 'When this status change occurred'
                    },
                    'notes': {
                        'type': 'string',
                        'required': False,
                        'description': 'Additional notes about this status change',
                        'maxlength': 2000
                    },
                    'changed_by': {
                        'type': 'string',
                        'required': False,
                        'description': 'Who initiated this status change (user_id or system)',
                        'maxlength': 100
                    }
                }
            }
        },
        'applied_at': {
            'type': 'date',
            'required': True,
            'description': 'When the application was officially submitted'
        },
        'updated_at': {
            'type': 'date',
            'required': True,
            'description': 'When the application was last updated'
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional application metadata',
            'properties': {
                'application_source': {'type': 'string'},
                'referral_source': {'type': 'string'},
                'salary_expectation': {
                    'type': 'object',
                    'properties': {
                        'amount': {'type': 'number'},
                        'currency': {'type': 'string'},
                        'period': {'type': 'string'}
                    }
                },
                'custom_fields': {'type': 'object'}
            }
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'applications',
        'indexes': [
            # Single field indexes
            {'key': [('user_id', 1)]},
            {'key': [('job_offer_id', 1)]},
            {'key': [('status', 1)]},
            {'key': [('applied_at', -1)]},
            {'key': [('updated_at', -1)]},
            
            # Compound indexes
            {
                'key': [
                    ('user_id', 1),
                    ('job_offer_id', 1)
                ],
                'unique': True
            },
            {
                'key': [
                    ('status', 1),
                    ('updated_at', -1)
                ]
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class Notification:
    """
    Schema for Notification documents.
    Tracks all system and user notifications within the platform.
    """
    schema = {
        '_id': Optional[str],  # MongoDB generated ObjectId
        'user_id': {
            'type': str,
            'required': True,
            'description': 'ID of the user who should receive this notification',
            'minlength': 1,
            'maxlength': 100
        },
        'type': {
            'type': 'string',
            'required': True,
            'description': 'Type/category of the notification',
            'enum': [
                'application_update', 'job_match', 'message_received', 'deadline_reminder',
                'profile_view', 'new_job_alert', 'system_alert', 'recommendation'
            ]
        },
        'title': {
            'type': 'string',
            'required': True,
            'description': 'Short title/heading of the notification',
            'maxlength': 200
        },
        'message': {
            'type': 'string',
            'required': True,
            'description': 'Full notification content',
            'maxlength': 2000
        },
        'related_entity': {
            'type': 'object',
            'required': False,
            'description': 'Entity this notification is related to',
            'properties': {
                'entity_type': {
                    'type': 'string',
                    'required': True,
                    'enum': ['job_offer', 'application', 'message', 'user', 'system']
                },
                'entity_id': {
                    'type': 'string',
                    'required': True,
                    'minlength': 1,
                    'maxlength': 100
                },
                'action': {
                    'type': 'string',
                    'required': False,
                    'description': 'Action related to the entity (e.g., created, updated, deleted)'
                }
            }
        },
        'read': {
            'type': 'boolean',
            'required': True,
            'default': False,
            'description': 'Whether the notification has been read by the user'
        },
        'read_at': {
            'type': 'date',
            'required': False,
            'description': 'When the notification was marked as read'
        },
        'priority': {
            'type': 'string',
            'required': True,
            'enum': ['low', 'medium', 'high', 'critical'],
            'default': 'medium',
            'description': 'Notification priority level'
        },
        'expires_at': {
            'type': 'date',
            'required': False,
            'description': 'When this notification should be automatically archived'
        },
        'created_at': {
            'type': 'date',
            'required': True,
            'description': 'When the notification was created'
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional notification metadata',
            'properties': {
                'source': {'type': 'string'},
                'tags': {'type': 'array', 'items': {'type': 'string'}},
                'custom_data': {'type': 'object'}
            }
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'notifications',
        'indexes': [
            # Single field indexes
            {'key': [('user_id', 1)]},
            {'key': [('type', 1)]},
            {'key': [('read', 1)]},
            {'key': [('priority', 1)]},
            {'key': [('created_at', -1)]},
            
            # Compound indexes
            {
                'key': [
                    ('user_id', 1),
                    ('read', 1),
                    ('created_at', -1)
                ]
            },
            {
                'key': [
                    ('related_entity.entity_type', 1),
                    ('related_entity.entity_id', 1)
                ],
                'sparse': True
            },
            {
                'key': [
                    ('expires_at', 1)
                ],
                'expireAfterSeconds': 0  # TTL index
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class ActivityLog:
    """Schema for tracking user and system activities for auditing and analytics."""
    schema = {
        '_id': Optional[str],
        'user_id': {'type': str, 'required': False, 'minlength': 1, 'maxlength': 100},
        'action': {'type': str, 'required': True, 'minlength': 1, 'maxlength': 100},
        'entity_type': {
            'type': 'string',
            'required': True,
            'enum': ['user', 'job_offer', 'application', 'cv', 'company', 'notification', 'system']
        },
        'entity_id': {'type': str, 'required': False, 'minlength': 1, 'maxlength': 100},
        'details': {'type': 'object', 'required': False, 'additionalProperties': True},
        'status': {
            'type': 'string',
            'required': True,
            'enum': ['success', 'failure', 'pending', 'error'],
            'default': 'success'
        },
        'timestamp': {'type': 'date', 'required': True},
        'ip_address': {'type': 'string', 'required': False, 'format': 'ip-address'},
        'user_agent': {'type': 'string', 'required': False, 'maxlength': 500},
        'metadata': {'type': 'object', 'required': False}
    }
    
    collection_config = {
        'collection_name': 'activity_logs',
        'indexes': [
            {'key': [('user_id', 1)]},
            {'key': [('action', 1)]},
            {'key': [('entity_type', 1), ('entity_id', 1)]},
            {'key': [('timestamp', -1)]},
            {
                'key': [('timestamp', 1)],
                'expireAfterSeconds': 60 * 60 * 24 * 30 * 6  # 6 months TTL
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error'
    }

class ChatHistory:
    """
    Schema for Chat History documents.
    Stores conversation history between users and the system/assistant.
    """
    schema = {
        '_id': Optional[str],  # MongoDB generated ObjectId
        'session_id': {
            'type': 'string',
            'required': True,
            'description': 'Unique identifier for the chat session',
            'minlength': 1,
            'maxlength': 100
        },
        'user_id': {
            'type': 'string',
            'required': True,
            'description': 'Reference to the user who owns this chat',
            'minlength': 1,
            'maxlength': 100
        },
        'title': {
            'type': 'string',
            'required': False,
            'description': 'Optional title for the conversation',
            'maxlength': 200
        },
        'chat_history': {
            'type': 'array',
            'required': True,
            'description': 'Array of message objects in the conversation',
            'minItems': 1,
            'items': {
                'type': 'object',
                'required': ['role', 'content', 'timestamp'],
                'properties': {
                    'role': {
                        'type': 'string',
                        'enum': ['user', 'assistant', 'system'],
                        'description': 'The role of the message sender'
                    },
                    'content': {
                        'type': 'string',
                        'description': 'The actual message content',
                        'minlength': 1,
                        'maxlength': 10000
                    },
                    'timestamp': {
                        'type': 'date',
                        'description': 'When the message was sent'
                    },
                    'metadata': {
                        'type': 'object',
                        'required': False,
                        'description': 'Additional message metadata',
                        'properties': {
                            'model': {'type': 'string'},
                            'tokens': {'type': 'number', 'minimum': 0},
                            'is_flagged': {'type': 'boolean'}
                        }
                    }
                }
            }
        },
        'created_at': {
            'type': 'date',
            'required': True,
            'description': 'When the chat session was created'
        },
        'updated_at': {
            'type': 'date',
            'required': True,
            'description': 'When the chat was last updated'
        },
        'is_active': {
            'type': 'boolean',
            'required': True,
            'default': True,
            'description': 'Whether the chat session is currently active'
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional chat session metadata',
            'properties': {
                'tags': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'uniqueItems': True
                },
                'source': {'type': 'string'},
                'custom_data': {'type': 'object'}
            }
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'chat_history',
        'indexes': [
            # Single field indexes
            {'key': [('user_id', 1)]},
            {'key': [('session_id', 1)], 'unique': True},
            {'key': [('created_at', -1)]},
            {'key': [('updated_at', -1)]},
            {'key': [('is_active', 1)]},
            
            # Compound indexes
            {
                'key': [
                    ('user_id', 1),
                    ('updated_at', -1)
                ]
            },
            {
                'key': [
                    ('user_id', 1),
                    ('is_active', 1)
                ]
            },
            # Text index for search
            {
                'key': [('chat_history.content', 'text')],
                'weights': {'chat_history.content': 1},
                'default_language': 'english'
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error',
        'capped': False,
        'size': None,  # No size limit for now
        'max': None    # No document limit for now
    }

class ChatMemory:
    """
    Schema for Chat Memory documents.
    Stores conversation context and memory for AI assistants to maintain continuity.
    """
    schema = {
        '_id': Optional[str],  # MongoDB generated ObjectId
        'key': {
            'type': 'string',
            'required': True,
            'description': 'Unique identifier for the memory context (usually user_id or session_id)',
            'minlength': 1,
            'maxlength': 100
        },
        'context': {
            'type': 'object',
            'required': False,
            'description': 'Contextual information to maintain conversation state',
            'properties': {
                'last_interaction': {'type': 'date'},
                'conversation_summary': {'type': 'string'},
                'user_preferences': {'type': 'object'},
                'conversation_stage': {'type': 'string'}
            }
        },
        'chat_history': {
            'type': 'array',
            'required': False,
            'description': 'Recent conversation history for context',
            'maxItems': 50,  # Limit to prevent unbounded growth
            'items': {
                'type': 'object',
                'required': ['role', 'content', 'timestamp'],
                'properties': {
                    'role': {
                        'type': 'string',
                        'enum': ['user', 'assistant', 'system'],
                        'description': 'The role of the message sender'
                    },
                    'content': {
                        'type': 'string',
                        'description': 'The message content',
                        'minlength': 1,
                        'maxlength': 5000
                    },
                    'timestamp': {
                        'type': 'date',
                        'description': 'When the message was sent'
                    },
                    'tokens': {
                        'type': 'number',
                        'minimum': 0,
                        'description': 'Number of tokens in the message'
                    }
                }
            }
        },
        'knowledge_snippets': {
            'type': 'array',
            'required': False,
            'description': 'Relevant knowledge or facts for this conversation',
            'items': {
                'type': 'object',
                'required': ['content', 'relevance'],
                'properties': {
                    'content': {'type': 'string'},
                    'source': {'type': 'string'},
                    'relevance': {
                        'type': 'number',
                        'minimum': 0,
                        'maximum': 1,
                        'description': 'Relevance score (0-1)'
                    },
                    'last_used': {'type': 'date'}
                }
            }
        },
        'created_at': {
            'type': 'date',
            'required': True,
            'description': 'When the memory was first created'
        },
        'updated_at': {
            'type': 'date',
            'required': True,
            'description': 'When the memory was last updated'
        },
        'ttl': {
            'type': 'date',
            'required': False,
            'description': 'Optional TTL for this memory (auto-delete after this time)'
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional metadata for the memory',
            'properties': {
                'version': {'type': 'string'},
                'tags': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'uniqueItems': True
                },
                'custom_data': {'type': 'object'}
            }
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'chat_memories',
        'indexes': [
            # Single field indexes
            {'key': [('key', 1)], 'unique': True},
            {'key': [('updated_at', -1)]},
            {'key': [('ttl', 1)], 'expireAfterSeconds': 0},  # TTL index
            
            # Compound indexes
            {
                'key': [
                    ('key', 1),
                    ('updated_at', -1)
                ]
            },
            # Text index for searching content
            {
                'key': [
                    ('chat_history.content', 'text'),
                    ('knowledge_snippets.content', 'text')
                ],
                'weights': {
                    'chat_history.content': 2,
                    'knowledge_snippets.content': 1
                },
                'default_language': 'english',
                'name': 'content_search'
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error',
        'capped': False
    }

class AgentTestQuery:
    """
    Schema for Agent Test Query documents.
    Stores test queries and their metadata for agent testing and analysis.
    """
    schema = {
        '_id': Optional[str],  # MongoDB generated ObjectId
        'job_title': {
            'type': 'string',
            'required': True,
            'description': 'Job title or position being queried',
            'minlength': 2,
            'maxlength': 200
        },
        'salary_expectation': {
            'type': 'string',
            'required': False,
            'description': 'Expected salary range (e.g., "35.000 - 45.000 EUR anuales")',
            'maxlength': 100
        },
        'location': {
            'type': 'string',
            'required': False,
            'description': 'Preferred work location (e.g., "España o remoto")',
            'maxlength': 200
        },
        'work_modality': {
            'type': 'string',
            'required': False,
            'description': 'Preferred work modality (e.g., "Híbrido", "Remoto", "Presencial")',
            'maxlength': 50
        },
        'skills': {
            'type': 'array',
            'required': False,
            'description': 'List of relevant skills or technologies',
            'items': {
                'type': 'string',
                'minlength': 2,
                'maxlength': 100
            }
        },
        'experience_level': {
            'type': 'string',
            'required': False,
            'enum': ['trainee', 'junior', 'mid', 'senior', 'lead', 'manager', 'director'],
            'description': 'Experience level for the position'
        },
        'timestamp': {
            'type': 'date',
            'required': True,
            'description': 'When the test query was executed'
        },
        'session_id': {
            'type': 'string',
            'required': True,
            'description': 'Session identifier to group related test queries',
            'minlength': 1,
            'maxlength': 100
        },
        'test_scenario': {
            'type': 'string',
            'required': False,
            'description': 'Identifier for different test scenarios',
            'maxlength': 100
        },
        'metadata': {
            'type': 'object',
            'required': False,
            'description': 'Additional metadata about the test query',
            'properties': {
                'user_agent': {
                    'type': 'string',
                    'maxlength': 500
                },
                'ip_address': {
                    'type': 'string',
                    'format': 'ip-address'
                },
                'test_parameters': {
                    'type': 'object',
                    'description': 'Additional parameters used for testing'
                },
                'response_metrics': {
                    'type': 'object',
                    'description': 'Performance and quality metrics for the test',
                    'properties': {
                        'response_time_ms': {'type': 'number', 'minimum': 0},
                        'result_count': {'type': 'integer', 'minimum': 0},
                        'success': {'type': 'boolean'}
                    }
                }
            }
        }
    }
    
    # MongoDB collection configuration
    collection_config = {
        'collection_name': 'agent_test_queries',
        'indexes': [
            # Single field indexes
            {'key': [('session_id', 1)], 'background': True},
            {'key': [('timestamp', -1)], 'background': True},
            {'key': [('test_scenario', 1)], 'background': True, 'sparse': True},
            
            # Compound indexes
            {
                'key': [
                    ('session_id', 1),
                    ('timestamp', -1)
                ],
                'background': True
            },
            {
                'key': [
                    ('test_scenario', 1),
                    ('timestamp', -1)
                ],
                'background': True,
                'sparse': True
            },
            # Text index for full-text search
            {
                'key': [
                    ('job_title', 'text'),
                    ('location', 'text'),
                    ('skills', 'text')
                ],
                'weights': {
                    'job_title': 3,
                    'location': 2,
                    'skills': 1
                },
                'default_language': 'spanish',
                'name': 'test_query_search',
                'background': True
            }
        ],
        'validation_level': 'strict',
        'validation_action': 'error',
        'capped': False
    }

# Collections
users = db['users']
job_sources = db['job_sources']
job_offers = db['job_offers']
cv_uploads = db['cv_uploads']
cv_rewrites = db['cv_rewrites']
job_matches = db['job_matches']
applications = db['applications']
notifications = db['notifications']
activity_logs = db['activity_logs']
chat_history = db['chat_history']  # Keep old collection
chat_memory = db['chat_memory']  # Add new collection
agent_test_queries = db['agent_test_queries']
