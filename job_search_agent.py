import os
from pymongo import MongoClient
from dotenv import load_dotenv

class JobSearchAgent:
    def __init__(self):
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            print("Error: MONGO_URI no encontrada en el archivo .env")
            print("Añade la línea: MONGO_URI=mongodb://localhost:27017/")
            exit()

        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping') # Check connection
            self.db = self.client.offheadhunter_db
            self.profiles_collection = self.db.agent_test_queries
            print("¡Conexión con MongoDB exitosa!")
        except Exception as e:
            print(f"Error al conectar con MongoDB: {e}")
            exit()

        self.user_id = "local_user_profile"  # Fixed ID for this single-user app
        self.user_profile = {}
        self.questions_map = {
            "job_title": "¿A qué posición te gustaría aplicar?",
            "salary_expectation": "¿Cuáles son tus expectativas salariales? (Indica el sueldo bruto anual y la moneda. Ejemplo: 30.000 EUR)",
            "location": "¿En qué país o zona te gustaría trabajar? (Puedes indicar solo el país, o también región y ciudad si lo deseas)",
            "work_modality": "¿En qué modalidad prefieres trabajar? (Presencial, Híbrido o A distancia)"
        }

    def load_profile(self):
        profile_data = self.profiles_collection.find_one({"_id": self.user_id})
        if profile_data:
            self.user_profile = profile_data
            if not self.is_profile_complete():
                print("¡Hola de nuevo! Continuemos creando tu perfil.")
        else:
            print("¡Hola! Soy tu asistente laboral inteligente para OffHeadHunter.")
            self.user_profile = {"_id": self.user_id}

    def save_profile(self):
        profile_data_to_save = self.user_profile.copy()
        self.profiles_collection.update_one(
            {"_id": self.user_id},
            {"$set": profile_data_to_save},
            upsert=True
        )

    def is_profile_complete(self):
        return all(key in self.user_profile and self.user_profile.get(key) for key in self.questions_map)

    def run(self):
        self.load_profile()

        while not self.is_profile_complete():
            for key, question in self.questions_map.items():
                if not self.user_profile.get(key):
                    answer = input(f"\n{question}\n> ")
                    self.user_profile[key] = answer.strip()
                    self.save_profile()
                    break

        print("\n" + "-"*40)
        print("Resumen de tu Perfil de Búsqueda:")
        print("-"*40)
        display_keys = {
            "job_title": "Cargo deseado",
            "salary_expectation": "Expectativa salarial",
            "location": "Ubicación",
            "work_modality": "Modalidad de trabajo"
        }
        for key, display_name in display_keys.items():
            print(f"🔹 {display_name}: {self.user_profile.get(key, 'No especificado')}")
        print("-"*40)
        
        print("\nYa hemos recopilado toda la información necesaria, ahora nos pondremos manos a la obra.")

if __name__ == '__main__':
    agent = JobSearchAgent()
    agent.run()
