import os
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import PyPDF2
from docx import Document
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition

class TextProcessor:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Inicializa el procesador de texto con un modelo de embeddings.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers a utilizar.
                       Por defecto usa 'all-mpnet-base-v2' que genera vectores de 768 dimensiones.
                       Es un modelo de alta calidad que funciona bien para tareas de búsqueda semántica.
        """
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def extract_text_from_file(self, file_path: Union[str, Path]) -> str:
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
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separator: str = "\n"
    ) -> List[Dict[str, Any]]:
        """
        Divide el texto en fragmentos (chunks) de tamaño manejable para procesamiento.
        
        Args:
            text: Texto a dividir.
            chunk_size: Tamaño máximo de cada chunk en tokens.
            chunk_overlap: Número de tokens de superposición entre chunks consecutivos.
            separator: Carácter o cadena para unir los chunks.
            
        Returns:
            Lista de diccionarios con los chunks y sus metadatos.
        """
        # Tokenizar el texto
        tokens = self.tokenizer.encode(text, disallowed_special=())
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calcular el índice final del chunk actual
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Decodificar los tokens a texto
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Agregar el chunk a la lista
            chunks.append({
                "text": chunk_text,
                "start_token": start_idx,
                "end_token": end_idx - 1,
                "num_tokens": len(chunk_tokens)
            })
            
            # Si hemos llegado al final, terminar
            if end_idx == len(tokens):
                break
                
            # Mover el índice de inicio, teniendo en cuenta el solapamiento
            start_idx = end_idx - chunk_overlap
            
            # Asegurarse de que no retrocedemos
            if start_idx < end_idx - chunk_overlap:
                start_idx = end_idx - chunk_overlap
        
        return chunks
    
    def generate_embeddings(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings para uno o más textos.
        
        Args:
            texts: Texto o lista de textos a vectorizar.
            batch_size: Tamaño del lote para procesamiento por lotes.
            
        Returns:
            Array de numpy con los embeddings generados.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Generar embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Procesa un documento completo: extrae texto, lo divide en chunks y genera embeddings.
        
        Args:
            file_path: Ruta al archivo a procesar.
            chunk_size: Tamaño máximo de cada chunk en tokens.
            chunk_overlap: Número de tokens de superposición entre chunks consecutivos.
            
        Returns:
            Lista de diccionarios con los chunks, sus metadatos y embeddings.
        """
        # Extraer texto
        text = self.extract_text_from_file(file_path)
        
        # Dividir en chunks
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        
        # Extraer solo los textos para generar embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generar embeddings para todos los chunks
        embeddings = self.generate_embeddings(chunk_texts)
        
        # Añadir los embeddings a los chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
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
