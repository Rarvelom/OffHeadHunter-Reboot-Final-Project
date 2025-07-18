import os
import json
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.binary import Binary
from datetime import datetime
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
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            print("Error: MONGO_URI no encontrada en el archivo .env")
            print("Añade la línea: MONGO_URI=mongodb://localhost:27017/")
            exit()

        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client["offheadhunter_db"]
            # Usando las colecciones especificadas
            self.profiles_collection = self.db["agent_test_queries"]  # Para perfiles de usuario
            self.cv_uploads = self.db["cv_uploads"]  # Para CVs
            self.user_id = "local_user_profile"  # ID para pruebas locales
            self.user_profile = {}
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
                "question": "¿En qué modalidad prefieres trabajar? (Presencial, Híbrido o A distancia)",
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
                
                # Prepare CV document for MongoDB
                cv_document = {
                    'user_id': self.user_id,
                    'file_name': os.path.basename(cv_path),
                    'file_data': Binary(file_data),
                    'upload_date': datetime.utcnow(),
                    'file_type': file_ext[1:],  # Remove the dot
                    'file_size': len(file_data)
                }
                
                # Save to MongoDB
                result = self.cv_uploads.insert_one(cv_document)
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
            
            if next_question:
                while True:
                    user_input = input(f"\n{next_question['question']}\n> ").strip()
                    
                    if not user_input:
                        print("Por favor, proporciona una respuesta.")
                        continue
                        
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
            else:
                # All questions answered, show summary
                self._display_profile_summary()
                
                # Ask for CV upload
                print("\n¡Perfecto! Ahora necesitamos que subas tu currículum vitae (CV).")
                self._upload_cv()
                
                print("\n¡Gracias por completar tu perfil!")
                print("Ahora nos pondremos manos a la obra con tu búsqueda de empleo.")
                print("\n¡Gracias por usar el asistente de búsqueda de empleo de OffHeadHunter! ¡Buena suerte con tu búsqueda!")
                break

if __name__ == '__main__':
    agent = JobSearchAgent()
    agent.run()
