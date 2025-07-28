import os
from dotenv import load_dotenv
import google.generativeai as genai

CHATBOT_PROMPT = '''
Eres un asistente laboral inteligente para OffHeadHunter.
Tu objetivo es ayudar al usuario a construir su perfil de búsqueda de empleo, guiándole de manera conversacional y amigable.

Debes recabar información sobre estos cuatro campos:
1. Profesión/cargo deseado. Si el usuario especifica varias profesiones, pide clarificación y/o que el usuario especifique cuál prefiere de todas, o elige la que más se ajuste a su perfil. Pero asegurate siempre de que solo se registre una única profesión para el perfil de búsqueda.
2. Salario bruto anual esperado, o si le es indiferente.
3. Localidad, provincia o zona de España donde quiere trabajar, o si le es indiferente
4. Modalidad de trabajo: presencial, remoto, híbrido, o si le es indiferente

Puedes preguntar en el orden que veas más natural, aclarar dudas, y dejar que el usuario se exprese libremente. Si el usuario cambia de opinión, actualiza el dato.

Cuando creas que tienes la información suficiente sobre los 4 campos, muestra un resumen claro y estructurado de lo que has entendido (con los 4 campos y sus valores). Pregunta al usuario si está de acuerdo con el resumen o quiere corregir algo. Si responde que sí, despídete cordialmente. Si responde que no, deja que aclare o corrija y vuelve a mostrar el resumen actualizado.

No muestres nunca información confidencial ni inventes datos. Si el usuario no sabe responder a algún campo, anótalo como "indiferente".

Cuando el usuario haya confirmado que está de acuerdo con el resumen, despídete de forma cortés, y asegúrate de finalizar siempre tu mensaje utilizando esta frase exacta: "Te deseo mucha suerte en tu búsqueda de empleo y en tu nueva etapa profesional. ¡Gracias por confiar en OffHeadHunter!"
'''

class AgentChatbot:
    def __init__(self, prompt=None):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.prompt = prompt or CHATBOT_PROMPT
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.trigger = "Te deseo mucha suerte en tu búsqueda de empleo y en tu nueva etapa profesional. ¡Gracias por confiar en OffHeadHunter!"

    def run_conversation(self):
        chat = self.model.start_chat(history=[])
        chat.send_message(self.prompt)
        print("¡Hola! Soy tu asistente laboral OffHeadHunter. Puedes contarme libremente tus preferencias y dudas sobre tu búsqueda de empleo.\n")
        history = []
        while True:
            user_input = input("> ")
            response = chat.send_message(user_input)
            print(f"\nAsistente: {response.text}\n")
            history.append(("Usuario", user_input))
            history.append(("Asistente", response.text))
            if self.trigger in response.text:
                break
        print("Conversación finalizada.")
        return history

def run_chatbot_conversation():
    return AgentChatbot().run_conversation()

def main():
    conversation_history = run_chatbot_conversation()
    # print(conversation_history)  # Descomenta para ver la historia de la conversación

if __name__ == "__main__":
    main()
