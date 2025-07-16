import os
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent))

from src.text_processing import TextProcessor
from src.qdrant_storage import QdrantStorage

# Cargar variables de entorno
load_dotenv()

def process_csv_file(
    file_path: Union[str, Path],
    collection_name: str = "cv_embeddings",
    user_id: str = None,
    text_column: str = "Resume",
    metadata_columns: List[str] = ["Category"],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Procesa un archivo CSV con CVs y almacena los embeddings en Qdrant.
    
    Args:
        file_path: Ruta al archivo CSV.
        collection_name: Nombre de la colección de Qdrant.
        user_id: ID del usuario propietario de los documentos.
        text_column: Nombre de la columna que contiene el texto del CV.
        metadata_columns: Lista de columnas a incluir como metadatos.
        chunk_size: Tamaño máximo de cada chunk en tokens.
        chunk_overlap: Número de tokens de superposición entre chunks.
        batch_size: Tamaño del lote para procesar los embeddings.
        
    Returns:
        Diccionario con información sobre el procesamiento.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"success": False, "error": f"El archivo {file_path} no existe."}
    
    try:
        # Inicializar el procesador de texto y el almacenamiento
        processor = TextProcessor()
        storage = QdrantStorage(collection_name=collection_name)
        
        # Leer el archivo CSV
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return {"success": False, "error": f"El archivo {file_path} está vacío o no contiene datos válidos."}
        
        # Verificar que existe la columna de texto
        if text_column not in rows[0]:
            return {"success": False, "error": f"La columna de texto '{text_column}' no existe en el archivo CSV."}
        
        total_chunks = 0
        processed_rows = 0
        
        # Procesar cada fila del CSV
        for row in rows:
            try:
                # Extraer el texto principal
                text = row[text_column].strip()
                if not text:
                    print(f"Advertencia: Fila {processed_rows + 1} tiene texto vacío. Se omite.")
                    continue
                
                # Extraer metadatos
                metadata = {col: row.get(col, "") for col in metadata_columns if col in row}
                
                # Procesar el texto en chunks
                chunks = processor.chunk_text(
                    text=text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                if not chunks:
                    print(f"Advertencia: No se pudieron extraer chunks de la fila {processed_rows + 1}.")
                    continue
                
                # Generar embeddings para los chunks
                chunk_texts = [chunk["text"] for chunk in chunks]
                embeddings = processor.generate_embeddings(chunk_texts, batch_size=batch_size)
                
                # Añadir los embeddings a los chunks
                for i, chunk in enumerate(chunks):
                    chunk["embedding"] = embeddings[i].tolist()
                
                # Crear un ID único para el documento
                doc_id = f"cv_{processed_rows}_{os.urandom(4).hex()}"
                
                # Almacenar los chunks en Qdrant
                stored_ids = storage.store_embeddings(
                    document_id=doc_id,
                    chunks=chunks,
                    metadata={
                        "source": file_path.name,
                        "row_index": processed_rows,
                        **metadata
                    },
                    user_id=user_id
                )
                
                total_chunks += len(stored_ids)
                processed_rows += 1
                
                if processed_rows % 10 == 0:
                    print(f"Procesadas {processed_rows} filas, {total_chunks} chunks almacenados...")
                
            except Exception as e:
                print(f"Error procesando la fila {processed_rows + 1}: {str(e)}")
                continue
        
        return {
            "success": True,
            "processed_rows": processed_rows,
            "total_chunks": total_chunks,
            "collection_name": collection_name,
            "file_path": str(file_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error al procesar el archivo CSV: {str(e)}",
            "file_path": str(file_path)
        }

def main():
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Procesar documentos y almacenar sus embeddings en Qdrant.")
    parser.add_argument("path", help="Ruta al archivo o directorio a procesar")
    parser.add_argument("--collection", "-c", default="cv_embeddings", 
                       help="Nombre de la colección de Qdrant (por defecto: cv_embeddings)")
    parser.add_argument("--user-id", "-u", default=None, 
                       help="ID del usuario propietario de los documentos")
    parser.add_argument("--text-column", default="Resume",
                       help="Nombre de la columna que contiene el texto (para archivos CSV)")
    parser.add_argument("--metadata-columns", nargs="+", default=["Category"],
                       help="Columnas a incluir como metadatos (para archivos CSV)")
    parser.add_argument("--chunk-size", type=int, default=800,
                       help="Tamaño máximo de cada chunk en tokens (por defecto: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=100,
                       help="Número de tokens de superposición entre chunks (por defecto: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Tamaño del lote para procesar los embeddings (por defecto: 32)")
    
    args = parser.parse_args()
    
    # Procesar la ruta (archivo o directorio)
    path = Path(args.path)
    if not path.exists():
        print(f"Error: La ruta {path} no existe.")
        sys.exit(1)
    
    # Determinar si es un archivo CSV
    if path.is_file() and path.suffix.lower() == '.csv':
        result = process_csv_file(
            file_path=path,
            collection_name=args.collection,
            user_id=args.user_id,
            text_column=args.text_column,
            metadata_columns=args.metadata_columns,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size
        )
        
        if result["success"]:
            print("\n" + "="*50)
            print("Procesamiento completado con éxito!")
            print(f"Archivo: {result['file_path']}")
            print(f"Filas procesadas: {result['processed_rows']}")
            print(f"Total de chunks almacenados: {result['total_chunks']}")
            print(f"Colección: {result['collection_name']}")
            print("="*50)
        else:
            print(f"\nError: {result.get('error', 'Error desconocido')}")
            sys.exit(1)
    else:
        print("Este script actualmente solo soporta archivos CSV.")
        print("Para otros tipos de archivos, use el script original process_documents.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
