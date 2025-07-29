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
    def split_by_sections(self, text: str) -> list:
        """
        Divide el texto en secciones usando encabezados típicos de CVs/job offers.
        Retorna una lista de dicts: [{'section': <nombre>, 'text': <texto>}]
        """
        # Encabezados comunes en español e inglés
        section_headers = [
            r"(?:^|\n)[ \t]*((experiencia laboral|experiencia profesional|experience|work experience|professional experience)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((educaci[oó]n|formaci[oó]n|education|academic background)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((habilidades|skills|competencias)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((resumen|summary|profile|perfil profesional|about me)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((proyectos|projects)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((idiomas|languages)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((certificaciones|certifications)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((publicaciones|publications)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((referencias|references)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((logros|achievements)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((intereses|interests)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((objetivo|objective)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((datos personales|personal data|contacto|contact information)[^\n:]*[:\n])",
            r"(?:^|\n)[ \t]*((informaci[oó]n adicional|additional information)[^\n:]*[:\n])",
        ]
        # Unir todos los patrones en uno solo
        pattern = '|'.join(section_headers)
        matches = [m for m in re.finditer(pattern, text, re.IGNORECASE)]
        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            section_title = match.group(1).strip().replace('\n', ' ').replace(':', '').strip() if match.group(1) else 'Seccion'
            section_text = text[start:end].strip()
            sections.append({'section': section_title, 'text': section_text})
        if not sections:
            sections = [{'section': 'completo', 'text': text}]
        return sections


    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        """
        Inicializa el procesador de texto con un modelo de embeddings.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers a utilizar.
                       Por defecto usa 'BAAI/bge-m3' que genera vectores de 1024 dimensiones.
                       Es un modelo optimizado para búsqueda semántica y generación de embeddings.
        """
        # Configurar para usar CPU
        self.model = SentenceTransformer(model_name, device='cpu')
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto de artefactos y ruido no deseado.
        """
        # Eliminar URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Eliminar correos electrónicos
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        # Eliminar números de teléfono (formatos variados)
        text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
        # Eliminar fechas (formatos simples, se puede expandir)
        text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
        # Eliminar caracteres especiales repetidos o aislados que no aportan significado
        text = re.sub(r'[^\w\s.,-]', ' ', text)
        # Eliminar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
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
                
            elif extension in ['.txt', '.md']:
                return file_path.read_text(encoding='utf-8')
                    
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
    
    def get_optimal_chunk_params(self, document_type: str = "cv") -> Dict[str, int]:
        """
        Obtiene parámetros optimizados de chunking según el tipo de documento.
        
        Args:
            document_type: Tipo de documento ('cv' o 'job')
            
        Returns:
            Diccionario con chunk_size y chunk_overlap optimizados
        """
        if document_type.lower() == "cv":
            return {
                "chunk_size": 1500,  # Más contexto para experiencias completas
                "chunk_overlap": 300,  # Mayor overlap para continuidad
                "min_section_tokens": 100  # Mínimo para considerar chunking
            }
        elif document_type.lower() == "job":
            return {
                "chunk_size": 1000,  # Suficiente para descripciones completas
                "chunk_overlap": 200,  # Overlap moderado
                "min_section_tokens": 80   # Mínimo para ofertas más cortas
            }
        else:
            # Valores por defecto
            return {
                "chunk_size": 1200,
                "chunk_overlap": 250,
                "min_section_tokens": 90
            }
    
    def smart_chunk_section(self, section_text: str, section_name: str, params: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Aplica chunking inteligente a una sección, respetando su semántica.
        
        Args:
            section_text: Texto de la sección
            section_name: Nombre de la sección
            params: Parámetros de chunking optimizados
            
        Returns:
            Lista de chunks con metadatos enriquecidos
        """
        # Tokenizar para evaluar tamaño
        tokens = self.tokenizer.encode(section_text, disallowed_special=())
        section_token_count = len(tokens)
        
        chunks = []
        
        # Si la sección es pequeña, mantenerla completa
        if section_token_count <= params["min_section_tokens"]:
            chunks.append({
                "text": section_text,
                "section": section_name,
                "start_token": 0,
                "end_token": section_token_count - 1,
                "num_tokens": section_token_count,
                "is_complete_section": True,
                "chunk_index": 0,
                "total_chunks_in_section": 1
            })
            return chunks
        
        # Si la sección es mediana y cabe en un chunk, mantenerla completa
        if section_token_count <= params["chunk_size"]:
            chunks.append({
                "text": section_text,
                "section": section_name,
                "start_token": 0,
                "end_token": section_token_count - 1,
                "num_tokens": section_token_count,
                "is_complete_section": True,
                "chunk_index": 0,
                "total_chunks_in_section": 1
            })
            return chunks
        
        # Para secciones grandes, aplicar chunking con overlap
        token_chunks = self.chunk_text(
            section_text, 
            params["chunk_size"], 
            params["chunk_overlap"]
        )
        
        # Enriquecer chunks con metadatos de sección
        total_chunks = len(token_chunks)
        for i, chunk in enumerate(token_chunks):
            chunk.update({
                "section": section_name,
                "is_complete_section": False,
                "chunk_index": i,
                "total_chunks_in_section": total_chunks
            })
            chunks.append(chunk)
        
        return chunks
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        document_type: str = "cv",
        chunk_size: int = None, 
        chunk_overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Procesa un documento completo con chunking semántico inteligente.
        
        Args:
            file_path: Ruta al archivo
            document_type: Tipo de documento ('cv' o 'job') para optimización
            chunk_size: Tamaño de chunk personalizado (opcional)
            chunk_overlap: Overlap personalizado (opcional)
        """
        # 1. Obtener parámetros optimizados
        params = self.get_optimal_chunk_params(document_type)
        
        # Permitir override de parámetros si se especifican
        if chunk_size is not None:
            params["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            params["chunk_overlap"] = chunk_overlap
        
        # 2. Extraer y limpiar texto
        text = self.extract_text_from_file(file_path)
        cleaned_text = self.clean_text(text)

        # 3. Dividir en secciones semánticas
        sections = self.split_by_sections(cleaned_text)
        
        # 4. Aplicar chunking inteligente por sección
        all_chunks = []
        for section in sections:
            section_chunks = self.smart_chunk_section(
                section['text'], 
                section['section'], 
                params
            )
            all_chunks.extend(section_chunks)
        
        # 5. Si no hay secciones identificadas, procesar como documento completo
        if len(sections) == 1 and sections[0]['section'] == 'completo':
            # Aplicar chunking estándar al documento completo
            token_chunks = self.chunk_text(
                cleaned_text, 
                params["chunk_size"], 
                params["chunk_overlap"]
            )
            all_chunks = []
            for i, chunk in enumerate(token_chunks):
                chunk.update({
                    "section": "documento_completo",
                    "is_complete_section": False,
                    "chunk_index": i,
                    "total_chunks_in_section": len(token_chunks)
                })
                all_chunks.append(chunk)
            
        # 6. Extraer textos para embeddings
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        if not chunk_texts:
            return []

        # 7. Generar embeddings
        embeddings = self.generate_embeddings(chunk_texts)
        
        # 8. Añadir embeddings y metadatos finales
        for i, chunk in enumerate(all_chunks):
            chunk["embedding"] = embeddings[i].tolist()
            chunk["document_type"] = document_type
            chunk["chunk_strategy"] = "semantic_intelligent"
            
        return all_chunks


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
