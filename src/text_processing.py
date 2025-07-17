import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Union
import logging
import re
from pathlib import Path
import PyPDF2
from docx import Document
import tiktoken
import numpy as np

# Cargar variables de entorno
load_dotenv()

# Configurar la API key de Google
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("No se encontró GOOGLE_API_KEY en las variables de entorno")

genai.configure(api_key=GOOGLE_API_KEY)

class TextProcessor:
    def __init__(self, model_name: str = 'text-embedding-004'):
        """
        Inicializa el procesador de texto con el modelo de embeddings de Google Generative AI.
        
        Args:
            model_name: Nombre del modelo de embeddings de Google a utilizar.
                     Por defecto usa 'text-embedding-004' que es compatible con Gemini 2.5 Flash.
        """
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def extract_text_from_file(self, file_path: Path) -> str:
        """
        Extrae texto de un archivo (PDF, DOCX, o TXT).
        
        Args:
            file_path: Ruta al archivo a procesar.
            
        Returns:
            Texto extraído del archivo.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                # Usamos Unstructured para PDFs ya que maneja mejor diferentes formatos
                elements = partition(str(file_path))
                return "\n\n".join([str(el) for el in elements])
                
            elif extension == '.docx':
                doc = Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
                
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            else:
                raise ValueError(f"Formato de archivo no soportado: {extension}")
                
        except Exception as e:
            raise Exception(f"Error al extraer texto de {file_path}: {str(e)}")
    
    def generate_embeddings(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings para uno o más textos utilizando Google Generative AI.
        
        Args:
            texts: Texto o lista de textos a convertir en embeddings.
            batch_size: Tamaño del lote para procesamiento por lotes.
            
        Returns:
            Array de numpy con los embeddings generados.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=batch,
                    task_type="retrieval_document",
                    title="Document chunk"
                )
                batch_embeddings = [np.array(embedding['values']) for embedding in response['embedding']]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generando embeddings para el lote {i//batch_size + 1}: {str(e)}")
                # Añadir arrays de ceros del tamaño esperado para mantener la consistencia
                expected_dim = 768  # Dimensión estándar para text-embedding-004
                all_embeddings.extend([np.zeros(expected_dim)] * len(batch))
        
        return np.array(all_embeddings)
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Divide el texto en chunks con solapamiento.
        
        Args:
            text: Texto a dividir.
            chunk_size: Tamaño máximo de cada chunk en tokens.
            chunk_overlap: Número de tokens de solapamiento entre chunks.
            
        Returns:
            Lista de diccionarios con los chunks y sus metadatos.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'tokens': chunk_tokens,
                'num_tokens': len(chunk_tokens),
                'start_token': i,
                'end_token': min(i + chunk_size, len(tokens))
            })
            
            if i + chunk_size >= len(tokens):
                break
                
        return chunks
    
    def process_document(
        self, 
        file_path: Path, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Procesa un documento completo: extrae texto, lo divide en chunks y genera embeddings.
        
        Args:
            file_path: Ruta al archivo a procesar.
            chunk_size: Tamaño máximo de cada chunk en tokens.
            chunk_overlap: Número de tokens de solapamiento entre chunks.
            
        Returns:
            Lista de diccionarios con los chunks, sus metadatos y embeddings.
        """
        # Extraer texto
        text = self.extract_text_from_file(file_path)
        
        # Dividir el texto en chunks
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        
        # Generar embeddings para cada chunk
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # Añadir los embeddings a los chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]['embedding'] = embedding.tolist()
            
        return chunks


# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia del procesador
    processor = TextProcessor()
    
    # Procesar un documento de ejemplo
    example_file = "example.pdf"  # Cambiar por la ruta a un archivo real
    
    if os.path.exists(example_file):
        chunks = processor.process_document(example_file)
        
        print(f"Documento procesado en {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Mostrar solo los primeros 3 chunks
            print(f"\nChunk {i+1} (tokens: {chunk['num_tokens']}):")
            print(chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"])
            print(f"Dimensión del embedding: {len(chunk['embedding'])}")
    else:
        print(f"Archivo de ejemplo no encontrado: {example_file}")
        print("Por favor, proporcione una ruta de archivo válida.")
