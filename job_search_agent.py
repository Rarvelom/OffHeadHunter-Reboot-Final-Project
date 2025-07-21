import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
from bson.binary import Binary
import google.generativeai as genai

SYSTEM_PROMPT = """Actúa como un asistente laboral inteligente para OffHeadHunter.
Tu tarea es recopilar la información necesaria del usuario para iniciar automáticamente una búsqueda de empleo mediante scraping en portales laborales, basada en los criterios proporcionados.

Guía al usuario paso a paso con tono profesional y amigable.

🟩 Flujo de preguntas:
1. Cargo deseado
2. Expectativa salarial
3. Ubicación deseada
4. Modalidad de trabajo
5. Sube tu CV

Sigue este flujo estrictamente y no pases a la siguiente pregunta hasta que la actual esté respondida adecuadamente.

Solo cuando tengas todas las respuestas, muestra un resumen y despídete con el mensaje final."""

class LLMService:
    def __init__(self, model: str = "gemini-2.5-flash"):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
        self.chat = None
        self._initialize_chat()

    def _initialize_chat(self):
        """Initialize the chat with the system prompt."""
        try:
            self.chat = self.model.start_chat(history=[])
            # Send system prompt as the first message
            self.chat.send_message(SYSTEM_PROMPT)
        except Exception as e:
            print(f"Error al inicializar el chat: {e}")

    def get_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Get a response from the model based on user input and context."""
        try:
            if not user_input.strip():
                return "Por favor, proporciona una respuesta válida."
                
            # Add context to the prompt if provided
            prompt = user_input
            if context:
                context_str = "\n".join(f"{k}: {v if v else 'No especificado'}" 
                                      for k, v in context.items())
                prompt = f"Contexto actual:\n{context_str}\n\nUsuario: {user_input}"
            
            # Get response from the model
            response = self.chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            print(f"Error al obtener respuesta de Gemini: {e}")
            return "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, inténtalo de nuevo más tarde."

class JobSearchAgent:
    def __init__(self):
        load_dotenv()
        self.llm = LLMService()
        
        # Initialize MongoDB connection
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            print("Error: MONGO_URI no encontrada en el archivo .env")
            print("Añade la línea: MONGO_URI=mongodb://localhost:27017/")
            exit()

        try:
            self.client = MongoClient(mongo_uri)
            self.qdrant_url = os.getenv("QDRANT_URL")
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if not self.qdrant_url or not self.qdrant_api_key:
                print("QDRANT_URL and QDRANT_API_KEY must be set.")
            else:
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                )
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

            self.db = self.client["offheadhunter_db"]
            # Usando las colecciones especificadas
            self.profiles_collection = self.db["agent_test_queries"]  # Para perfiles de usuario
            self.cv_uploads = self.db["cv_uploads"]  # Para CVs
            # Generar un ID único para cada nuevo perfil
            self.user_id = f"user_{str(uuid.uuid4())}"  # ID único para cada perfil
            self.user_profile = {}
            print(f"ID de perfil generado: {self.user_id}")
            print("¡Conexión con MongoDB exitosa!")
            print(f"Base de datos: offheadhunter_db")
            print(f"Colección de perfiles: agent_test_queries")
            print(f"Colección de CVs: cv_uploads")
        except Exception as e:
            print(f"Error al conectar con MongoDB: {e}")
            exit()

        self.user_profile = {}
        self.questions = [
            {
                "key": "job_title",
                "question": "¿A qué posición te gustaría aplicar?",
                "description": "Cargo deseado"
            },
            {
                "key": "salary_expectation",
                "question": "¿Cuáles son tus expectativas salariales? (Indica el sueldo bruto anual y la moneda. Ejemplo: 30.000 EUR)",
                "description": "Expectativa salarial"
            },
            {
                "key": "location",
                "question": "¿En qué país o zona te gustaría trabajar? (Puedes indicar solo el país, o también región y ciudad si lo deseas)",
                "description": "Ubicación"
            },
            {
                "key": "work_modality",
                "question": "¿En qué modalidad prefieres trabajar? (Presencial, Híbrido, A distancia)",
                "description": "Modalidad de trabajo"
            }
        ]

    def load_profile(self, reset: bool = False):
        """Load user profile from MongoDB. If reset is True, start with empty profile."""
        if not reset:
            profile_data = self.profiles_collection.find_one({"_id": self.user_id})
            if profile_data:
                self.user_profile = profile_data
                if not self.is_profile_complete():
                    print("¡Hola de nuevo! Continuemos completando tu perfil.")
            else:
                print("¡Hola! Soy tu asistente laboral inteligente para OffHeadHunter.")
                self.user_profile = {"_id": self.user_id}
        else:
            # Start with empty profile
            self.user_profile = {
                "job_title": "",
                "salary_expectation": "",
                "location": "",
                "work_modality": ""
            }

    def save_profile(self):
        profile_data_to_save = self.user_profile.copy()
        self.profiles_collection.update_one(
            {"_id": self.user_id},
            {"$set": profile_data_to_save},
            upsert=True
        )

    def is_profile_complete(self):
        return all(q["key"] in self.user_profile and self.user_profile.get(q["key"]) for q in self.questions)

    def _get_next_question(self) -> Optional[Dict[str, str]]:
        """Get the next unanswered question."""
        for question in self.questions:
            if not self.user_profile.get(question["key"]):
                return question
        return None

    def _upload_cv(self):
        """Handle CV file upload and store it in MongoDB."""
        print("\n" + "="*60)
        print("SUBA SU CURRÍCULUM VITAE (CV)")
        print("="*60)
        print("\nPor favor, proporcione la ruta completa a su archivo CV (formato PDF o DOCX):")
        
        while True:
            cv_path = input("> ").strip()
            
            if not cv_path:
                print("Por favor, proporcione una ruta válida.")
                continue
                
            # Check if file exists
            if not os.path.isfile(cv_path):
                print(f"Error: No se encontró el archivo en la ruta: {cv_path}")
                print("Por favor, verifique la ruta e inténtelo de nuevo.")
                continue
                
            # Check file extension
            file_ext = os.path.splitext(cv_path)[1].lower()
            if file_ext not in ['.pdf', '.docx']:
                print("Error: Solo se admiten archivos PDF o DOCX.")
                continue
                
            try:
                # Read file as binary
                with open(cv_path, 'rb') as file:
                    file_data = file.read()
                
                # Extract text from file
                elements = partition(cv_path)
                cv_text = "\n".join([str(el) for el in elements])

                # Prepare CV document for MongoDB with all required fields
                cv_document = {
                    'user_id': self.user_id,
                    'filename': os.path.basename(cv_path),
                    'file_url': f"file://{cv_path}",  # Store the file path as URL
                    'original_text': cv_text,
                    'version': 1,  # Initial version
                    'vectorized': False,  # Will be updated after Qdrant save
                    'uploaded_at': datetime.utcnow(),
                    'status': 'pending',
                    'metadata': {
                        'file_size': len(file_data),
                        'file_type': file_ext[1:],  # Remove the dot
                        'pages': len(cv_text.split('\n'))  # Estimate pages
                    }
                }
                
                # Save to Qdrant first to get the vector ID
                filename = os.path.basename(cv_path)
                if hasattr(self, 'qdrant_client') and self.qdrant_client:
                    if not cv_text:
                        print("Warning: CV text is empty. Cannot save to Qdrant.")
                    else:
                        try:
                            # Generate embedding
                            embedding = self.embedding_model.encode(cv_text).tolist()
                            
                            # Generate a unique ID for Qdrant
                            cv_id = str(uuid.uuid4())
                            
                            # Save to Qdrant
                            self.qdrant_client.upsert(
                                collection_name="cv_embeddings",
                                points=[
                                    models.PointStruct(
                                        id=cv_id,
                                        vector=embedding,
                                        payload={
                                            "mongodb_id": "",  # Will be updated after MongoDB save
                                            "filename": filename,
                                            "text": cv_text
                                        }
                                    )
                                ],
                                wait=True
                            )
                            
                            # Update CV document with Qdrant info
                            cv_document['embedding_vector_id_qdrant'] = cv_id
                            cv_document['embedding_model'] = 'all-mpnet-base-v2'
                            cv_document['vectorized'] = True
                            
                            print(f"CV '{filename}' saved to Qdrant with id {cv_id}")
                            
                        except Exception as e:
                            print(f"Failed to save CV to Qdrant: {e}")
                            cv_document['status'] = 'error'
                            cv_document['error'] = str(e)
                else:
                    print("Qdrant client not available. Cannot save vector data.")
                    cv_document['status'] = 'error'
                    cv_document['error'] = 'Qdrant client not available'
                
                # Save to MongoDB with all fields including Qdrant info
                result = self.cv_uploads.insert_one(cv_document)
                
                # Update Qdrant with the MongoDB ID if the save was successful
                if cv_document.get('vectorized', False) and hasattr(self, 'qdrant_client') and self.qdrant_client:
                    try:
                        self.qdrant_client.set_payload(
                            collection_name="cv_embeddings",
                            payload={"mongodb_id": str(result.inserted_id)},
                            points=[cv_document['embedding_vector_id_qdrant']]
                        )
                    except Exception as e:
                        print(f"Warning: Could not update Qdrant with MongoDB ID: {e}")

                print("\n CV subido exitosamente a la base de datos.")
                return True
                
            except Exception as e:
                print(f"Error al procesar el archivo: {e}")
                return False

    def _display_profile_summary(self):
        """Display a summary of the collected profile information."""
        print("\n" + "-"*40)
        print("Resumen de tu Perfil de Búsqueda:")
        print("-"*40)
        for question in self.questions:
            value = self.user_profile.get(question["key"], 'No especificado')
            print(f" {question['description']}: {value}")
        print("-"*40)

    def run(self, reset_profile: bool = True):
        self.load_profile(reset=reset_profile)
        
        print("¡Hola! Soy tu asistente de búsqueda de empleo OffHeadHunter.\n")
        print("Voy a ayudarte a completar tu perfil de búsqueda de empleo.\n")
        
        # Ask profile questions
        while True:
            next_question = self._get_next_question()
            
            if next_question is None:
                # All questions answered, show summary
                self._display_profile_summary()
                
                # Ask for CV upload
                print("\n¡Perfecto! Ahora necesitamos que subas tu currículum vitae (CV).")
                self._upload_cv()
                
                print("\n¡Gracias por completar tu perfil!")
                print("Ahora nos pondremos manos a la obra con tu búsqueda de empleo.")
                print("\n¡Gracias por usar el asistente de búsqueda de empleo de OffHeadHunter! ¡Buena suerte con tu búsqueda!")
                break
                
            # Ask the current question
            while True:
                user_input = input(f"\n{next_question['question']}\n> ").strip()
                
                if not user_input:
                    print("Por favor, proporciona una respuesta.")
                    continue
                    
                # Special validation for work modality
                if next_question["key"] == "work_modality":
                    normalized_input = user_input.strip().capitalize()
                    if normalized_input in ["Presencial", "Híbrido", "A distancia"]:
                        llm_response = "VÁLIDO"
                        # Save the normalized version (capitalized)
                        user_input = normalized_input
                    else:
                        llm_response = "Por favor, elige una de las opciones: Presencial, Híbrido o A distancia"
                # Special validation for location
                elif next_question["key"] == "location":
                    # Check if the input looks like a valid location (letters, spaces, hyphens, and some special characters for cities)
                    if not user_input.replace(' ', '').replace('-', '').replace('.', '').replace("'", "").replace("´", "").isalpha():
                        llm_response = "Por favor, ingresa un nombre de país o ciudad válido (solo letras, espacios y guiones)."
                    else:
                        validation_prompt = (
                            f"¿Es '{user_input}' un nombre de país o ciudad válido? "
                            "Responde solo con 'VÁLIDO' si es un país o ciudad real, o con una explicación si no lo es. "
                            "No incluyas nada más en tu respuesta."
                        )
                        llm_response = self.llm.get_response(validation_prompt, self.user_profile)
                else:
                    # Standard validation for other questions
                    validation_prompt = (
                        f"El usuario respondió: '{user_input}' a la pregunta: '{next_question['question']}'. "
                        "¿Es una respuesta válida? Si no es clara, pide aclaraciones de manera amable. "
                        "Responde solo con 'VÁLIDO' si la respuesta es correcta, o con una explicación clara si necesita aclaración."
                    )
                    llm_response = self.llm.get_response(validation_prompt, self.user_profile)
                
                if llm_response.strip().upper() == "VÁLIDO":
                    self.user_profile[next_question["key"]] = user_input
                    self.save_profile()
                    break
                else:
                    print(f"\nAsistente: {llm_response}")

if __name__ == '__main__':
    print("--- Script execution started ---", flush=True)
    agent = JobSearchAgent()
    print("--- Agent initialized ---", flush=True)
    agent.run()
    print("--- Script execution finished ---", flush=True)
