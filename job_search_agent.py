import os
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from unstructured.partition.auto import partition
from bson.binary import Binary
import uuid
from datetime import datetime

SYSTEM_PROMPT = """
Eres un agente de análisis de conversaciones para OffHeadHunter.

Tu función es recibir el historial completo de una conversación entre un usuario y un asistente laboral, y extraer de ese historial SOLO la información relevante para construir un perfil de búsqueda de empleo estructurado.

Debes identificar y transformar los siguientes cuatro campos clave, devolviendo el resultado en un único bloque JSON válido, listo para ser procesado automáticamente:

1. job_title: Profesión clara y corta, útil como palabra clave para usar como parametro de búsqueda en una página web.
2. salary_expectation: Número entero que represente el salario bruto anual, sin puntos ni símbolos. Si es mensual, conviértelo a anual. Si es indiferente o no especifica nada concreto, pon 0.
3. location: Devuelve el/los código(s) numérico(s) de provincias de Esapaña según esta tabla:

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

    - Si el usuario no especifica ninguna ubicación concreta, indica que no le importa la localidad o que está abierto a cualquier ubicación (o responde "España") devuelve el código 0.
    

4. work_modality: Traduce a los códigos: Presencial=1, A distancia=2, Híbrido=3, Sin especificar=4. Puede ser varios (ej: "2,3"). Si selecciona una o dos modalidades como preferencia (ej: Presencial y Híbrido, o Remoto y Híbrido), incluye también la modalidad "Sin especificar" (4). Si dice que no tiene preferencia, le es indiferente, o no encaja con ninguno de los otros códigos, devuelve 0".

IMPORTANTE:
- Analiza toda la conversación para encontrar la información más actualizada y relevante sobre cada campo, aunque el usuario haya cambiado de opinión.
- Devuelve SOLO el bloque JSON, sin explicaciones ni texto adicional.
- Si algún campo no puede determinarse, asígnale el valor por defecto correspondiente (ej: 0 o 4 según el caso).

Ejemplo de respuesta:
{
  "job_title": "Desarrollador web",
  "salary_expectation": 30000,
  "location": "33,9",
  "work_modality": "1,3"
}
"""

class JobSearchAgent:
    def __init__(self):
        load_dotenv()
        # MongoDB
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
            self.profiles_collection = self.db["agent_test_queries"]
            self.cv_uploads = self.db["cv_uploads"]
            self.user_id = f"user_{str(uuid.uuid4())}"
            self.user_profile = {}
            print(f"ID de perfil generado: {self.user_id}")
            print("¡Conexión con MongoDB exitosa!")
            print(f"Base de datos: offheadhunter_db")
            print(f"Colección de perfiles: agent_test_queries")
            print(f"Colección de CVs: cv_uploads")
        except Exception as e:
            print(f"Error al conectar con MongoDB: {e}")
            exit()

    def parse_profile_from_text(self, history_text):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = SYSTEM_PROMPT + "\n\nCONVERSACIÓN:\n" + history_text
        response = model.generate_content(prompt)
        match = re.search(r"\{[\s\S]*?\}", response.text)
        if match:
            json_str = match.group(0)
            try:
                profile = json.loads(json_str)
                self.user_profile = profile
                return profile
            except Exception as e:
                print("No se pudo parsear el JSON extraído:", e)
                print("Respuesta completa del modelo:\n", response.text)
                return None
        else:
            print("No se detectó un bloque JSON en la respuesta del modelo.")
            print("Respuesta completa:\n", response.text)
            return None

    def save_profile(self):
        if self.user_profile:
            self.user_profile["user_id"] = self.user_id
            try:
                # Usar update_one con upsert=True para insertar si no existe o actualizar si ya existe.
                self.profiles_collection.update_one(
                    {"user_id": self.user_profile["user_id"]},
                    {"$set": self.user_profile},
                    upsert=True
                )
                print("Perfil guardado/actualizado en MongoDB.")
            except Exception as e:
                print(f"Error al guardar el perfil en MongoDB: {e}")

    def load_profile(self):
        profile = self.profiles_collection.find_one({"user_id": self.user_id})
        if profile:
            self.user_profile = profile
            print("Perfil cargado de MongoDB.")
        else:
            print("No se encontró un perfil con este user_id.")

    def upload_cv(self, file_path=None):
        """
        Procesa y sube el CV a MongoDB/Qdrant.
        - file_path: ruta al archivo, si None usa input() (modo CLI).
        """
        if file_path is None:
            print("\nPor favor, proporcione la ruta completa a su archivo CV:")
            while True:
                file_path = input("> ").strip()
                if file_path:
                    break

        cv_path = Path(file_path)
        if not cv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {cv_path}")

        file_ext = cv_path.suffix.lower()
        if file_ext not in ['.pdf', '.docx']:
            raise ValueError("Solo se admiten archivos PDF o DOCX.")

        with open(cv_path, "rb") as f:
            cv_data = f.read()

        # Extraer texto
        elements = partition(str(cv_path))
        cv_text = "\n".join([str(el) for el in elements])

        cv_id = str(uuid.uuid4())
        filename = cv_path.name
        uploaded_at = datetime.utcnow().isoformat()

        cv_record = {
            'user_id': self.user_id,
            'cv_id': cv_id,
            'filename': filename,
            'file_url': f"file://{cv_path}",
            'original_text': cv_text,
            'version': 1,
            'vectorized': False,
            'uploaded_at': uploaded_at,
            'status': 'pending',
            'file_binary': Binary(cv_data),
            'metadata': {
                'file_size': len(cv_data),
                'file_type': file_ext[1:],
                'pages': len(cv_text.split('\n'))
            }
        }

        # Guardar embedding en Qdrant
        if self.qdrant_client:
            embedding = self.embedding_model.encode(cv_text).tolist()
            self.qdrant_client.upsert(
                collection_name="cv_embeddings_BGE2",
                points=[models.PointStruct(
                    id=cv_id,
                    vector=embedding,
                    payload={"filename": filename, "text": cv_text}
                )]
            )
            cv_record['embedding_vector_id_qdrant'] = cv_id
            cv_record['vectorized'] = True

        self.cv_uploads.insert_one(cv_record)
        self.user_profile['cv_path'] = str(cv_path)
        self.save_profile()
        return str(cv_path)


def parse_profile_from_text(history_text):
    return JobSearchAgent().parse_profile_from_text(history_text)

def main():
    agent = JobSearchAgent()
    chat = agent.qdrant_client.start_chat(history=[])
    chat.send_message(SYSTEM_PROMPT)

    print("Introduce tus preferencias laborales en un solo mensaje (puedes escribirlo de forma natural):\n")
    user_input = input("> ")
    response = chat.send_message(user_input)
    # Busca el primer bloque JSON en la respuesta
    match = re.search(r"\{[\s\S]*?\}", response.text)
    if match:
        json_str = match.group(0)
        try:
            profile = json.loads(json_str)
            print("\nPerfil extraído:")
            print(json.dumps(profile, indent=2, ensure_ascii=False))
            agent.save_profile()
        except Exception as e:
            print("No se pudo parsear el JSON extraído:", e)
            print("Respuesta completa del modelo:\n", response.text)
    else:
        print("No se detectó un bloque JSON en la respuesta del modelo.")
        print("Respuesta completa:\n", response.text)

    agent.upload_cv()

if __name__ == "__main__":
    main()
