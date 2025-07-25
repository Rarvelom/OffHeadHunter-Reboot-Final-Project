import os
import json
import uuid
import re
from datetime import datetime, timezone, timedelta
from src.utils.time_utils import get_current_utc_timestamp
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
from bson.binary import Binary
import google.generativeai as genai

SYSTEM_PROMPT = """
Actúa como un asistente laboral inteligente para OffHeadHunter.

Tu tarea es guiar al usuario para completar un perfil de búsqueda laboral y transformar sus respuestas en un formato útil para scraping automático.

🔷 Para cada campo, debes asegurarte de lo siguiente:

1. **job_title**: 
   - Extrae una profesión clara en formato corto, útil como palabra clave. 
   - Ejemplo: De "Me gustaría trabajar ayudando a personas con problemas sociales", responde: "Trabajador social".

2. **salary_expectation**:
   - Extrae un número que represente el salario bruto anual en euros.
   - Convierte si es mensual: "1500€ al mes" → 18000.

3. **location**:
   - Devuelve el/los código(s) numérico(s) correspondientes según esta tabla:
            {
                "A Coruña": 28,
                "Álava/Araba": 2,
                "Albacete": 3,
                "Alicante/Alacant": 4,
                "Almería": 5,
                "Asturias": 6,
                "Ávila": 7,
                "Badajoz": 8,
                "Barcelona": 9,
                "Burgos": 10,
                "Cáceres": 11,
                "Cádiz": 12,
                "Cantabria": 13,
                "Castellón/Castelló": 14,
                "Ceuta": 15,
                "Ciudad Real": 16,
                "Córdoba": 17,
                "Cuenca": 18,
                "Girona": 19,
                "Granada": 21,
                "Guadalajara": 22,
                "Guipúzcoa/Gipuzkoa": 23,
                "Huelva": 24,
                "Huesca": 25,
                "Islas Baleares/Illes Balears": 26,
                "Jaén": 27,
                "La Rioja": 29,
                "Las Palmas": 20,
                "León": 30,
                "Lleida": 31,
                "Lugo": 32,
                "Madrid": 33,
                "Málaga": 34,
                "Melilla": 35,
                "Murcia": 36,
                "Navarra": 37,
                "Ourense": 38,
                "Palencia": 39,
                "Pontevedra": 40,
                "Salamanca": 41,
                "Santa Cruz de Tenerife": 46,
                "Segovia": 42,
                "Sevilla": 43,
                "Soria": 44,
                "Tarragona": 45,
                "Teruel": 47,
                "Toledo": 48,
                "Valencia/València": 49,
                "Valladolid": 50,
                "Vizcaya/Bizkaia": 51,
                "Zamora": 52,
                "Zaragoza": 53
            }
   - Si no está en España, responde que solo se permiten ubicaciones españolas. 
   - Si no se especifica ninguna ubicación concreta, responde que no le importa la localidad o que está abierto a cualquier ubicación, deja este campo vacío y continua con la siguiente pregunta.
   - Si el usuario responde que quiere buscar en toda España, o solo España, deja este campo vacio y continua con la siguiente pregunta.

4. **work_modality**:
   - Traduce a los códigos siguientes:
     Presencial = 1, A distancia = 2, Híbrido = 3, Sin especificar = 4
   - Puede devolver varios valores (ej. "2,3").
   - Si la persona no especifica nada o dice que le es indiferente  o que está abierto a cualquier modalidad, escoge todas las opciones (1, 2, 3, 4).

Si consideras que necesitas mas información para cada campo, pide aclaraciones de manera amable, y no avances a la siguiente pregunta hasta que no hayas obtenido la información necesaria para completar el campo correctamente. En ese caso, quédate con la última respuesta dada
Siempre responde con la forma **transformada** y lista para ser almacenada, no repitas la entrada original del usuario.

Una vez todo esté recogido, muestra un resumen de lo recopilado y despídete con cortesía profesional.
"""

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
            self.embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')

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
                "question": "¿Cuáles son tus expectativas salariales? (Indica el sueldo bruto anual que desearías para tu cargo indicado. Por ejemplo: 30.000)",
                "description": "Expectativa salarial"
            },
            {
                "key": "location",
                "question": "¿En qué país o zona te gustaría trabajar? (Puedes indicar solo el país, o también región y ciudad si lo deseas)",
                "description": "Ubicación"
            },
            {
                "key": "work_modality",
                "question": "¿En qué modalidad prefieres trabajar? (Presencial, Híbrido, A distancia, Indiferente)",
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
                    'file_url': f"file://{cv_path}",  # URL del archivo
                    'original_text': cv_text,
                    'version': 1,  # Version inicial
                    'vectorized': False,  # Will be updated after Qdrant save
                    'uploaded_at': get_current_utc_timestamp(),  # Guardar el timestamp actual sin ajustes
                    'status': 'pending',
                    'file_binary': Binary(file_data),  # Binario original del CV (PDF o DOCX)
                    'metadata': {
                        'file_size': len(file_data),
                        'file_type': file_ext[1:],  # Tipo de archivo (PDF o DOCX), se extrae el punto de la extensión.
                        'pages': len(cv_text.split('\n'))  # Paginas estimadas
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
                                collection_name="cv_embeddings_BGE",
                                points=[
                                    models.PointStruct(
                                        id=cv_id,
                                        vector=embedding,
                                        payload={
                                            "mongodb_id": "",  # Will be updated after MongoDB save
                                            "filename": filename,
                                            "text": cv_text,
                                            "uploaded_at": get_current_utc_timestamp()  # Guardar el timestamp actual sin ajustes
                                        }
                                    )
                                ],
                                wait=True
                            )
                            
                            # Update CV document with Qdrant info
                            cv_document['embedding_vector_id_qdrant'] = cv_id
                            cv_document['embedding_model'] = 'BGE-m3'
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
                            collection_name="cv_embeddings_BGE",
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

    @staticmethod
    def parse_llm_response(key: str, raw_response: str) -> str:
        """
        Extrae y limpia el valor de una respuesta generada por el modelo LLM.
        
        Args:
            key (str): El nombre del campo esperado (ej: 'job_title').
            raw_response (str): La respuesta devuelta por Gemini.
        
        Returns:
            str: El valor limpio asociado a la clave esperada.
        """
        if not raw_response:
            return ""

        # Si la respuesta es muy larga, tomar solo la primera línea
        first_line = raw_response.strip().split('\n')[0].strip()
        
        # Eliminar cualquier texto después de un punto, signo de interrogación o exclamación
        # que pueda ser parte de una conversación
        for punct in ['.', '?', '!']:
            if punct in first_line:
                first_line = first_line.split(punct)[0].strip()
        
        # Eliminar comillas y otros caracteres especiales
        first_line = re.sub(r'["\']', '', first_line)
        
        # Eliminar cualquier prefijo común de conversación
        first_line = re.sub(r'^(?:\*\*)?(?:[A-Za-záéíóúÁÉÍÓÚñÑ\s]+\s*:\s*)?(?:\*\*)?', '', first_line)
        first_line = re.sub(r'^[\s\-\*]+', '', first_line)
        
        # Si después de la limpieza no queda nada, usar la respuesta original
        if not first_line.strip():
            first_line = raw_response.strip()
        
        # Limitar la longitud para evitar respuestas demasiado largas
        max_length = 100  # Longitud máxima razonable para un campo
        if len(first_line) > max_length:
            first_line = first_line[:max_length].rsplit(' ', 1)[0] + '...'
            
        return first_line.strip()

    def run(self, reset_profile: bool = True):
        self.load_profile(reset=reset_profile)
        
        print("¡Hola! Soy tu asistente de búsqueda de empleo OffHeadHunter.\n")
        print("Voy a ayudarte a completar tu perfil de búsqueda de empleo.\n")
        
        while True:
            next_question = self._get_next_question()
            
            if next_question is None:
                self._display_profile_summary()
                # print(self.user_profile)
                print("\n¡Perfecto! Ahora necesitamos que subas tu currículum vitae (CV).")
                self._upload_cv()
                print("\n¡Gracias por completar tu perfil!")
                print("Ahora nos pondremos manos a la obra con tu búsqueda de empleo.")
                print("\n¡Gracias por usar el asistente de búsqueda de empleo de OffHeadHunter! ¡Buena suerte con tu búsqueda!")
                break
            
            while True:
                user_input = input(f"\n{next_question['question']}\n> ").strip()
                
                if not user_input:
                    print("Por favor, proporciona una respuesta.")
                    continue

                question_text = next_question['question']
                transformation_prompt = (
                    f"Pregunta: {question_text}\n"
                    f"Respuesta del usuario: {user_input}\n"
                    f"Transforma esta respuesta según las reglas anteriores."
                )

                raw_response = str(self.llm.get_response(transformation_prompt, self.user_profile)).strip()
                cleaned_response = self.parse_llm_response(next_question["key"], raw_response)
                
                # Validación adicional según el tipo de campo
                if next_question["key"] == "job_title":
                    # Asegurar que el título del trabajo no esté vacío
                    if not cleaned_response or len(cleaned_response) < 2:
                        print("Por favor, proporciona un título de trabajo válido.")
                        continue
                    # Eliminar cualquier número o carácter especial al principio
                    cleaned_response = re.sub(r'^[\d\s\-\*]+', '', cleaned_response).strip()
                
                elif next_question["key"] == "salary_expectation":
                    # Extraer solo los números de la respuesta
                    numbers = re.findall(r'\d+', cleaned_response)
                    if numbers:
                        cleaned_response = numbers[0]  # Tomar el primer número encontrado
                    else:
                        print("Por favor, proporciona un valor numérico para el salario.")
                        continue
                
                elif next_question["key"] == "work_modality":
                    # Normalizar la modalidad de trabajo
                    modalidad = cleaned_response.lower()
                    if any(m in modalidad for m in ["presencial", "oficina"]):
                        cleaned_response = "Presencial"
                    elif any(m in modalidad for m in ["híbrido", "hibrido"]):
                        cleaned_response = "Híbrido"
                    elif any(m in modalidad for m in ["remoto", "distancia", "teletrabajo"]):
                        cleaned_response = "A distancia"
                    else:
                        print("Opción no reconocida. Usando 'Indiferente' como valor por defecto.")
                        cleaned_response = "Indiferente"
                
                # Guardar la respuesta limpia
                self.user_profile[next_question["key"]] = cleaned_response
                self.save_profile()
                break
            else:
                print("\nAsistente: No pude entender esa respuesta. ¿Podrías reformularla?")


if __name__ == '__main__':
    print("--- Script execution started ---", flush=True)
    agent = JobSearchAgent()
    print("--- Agent initialized ---", flush=True)
    agent.run()
    print("--- Script execution finished ---", flush=True)
