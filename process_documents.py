import os
import sys
import argparse
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.text_processing import TextProcessor
from src.pdf_processor import PDFProcessor
from src.qdrant_storage import QdrantStorage

# Load environment variables
load_dotenv()

def process_document_file(
    file_path: Union[str, Path],
    collection_name: str = "cv_embeddings",
    user_id: str = None,
    document_type: str = "cv",
    chunk_size: int = None,
    chunk_overlap: int = None,
    batch_size: int = 32,
    source: str = "cv_upload",
    text_processor: Optional[TextProcessor] = None,
    storage: Optional[QdrantStorage] = None
) -> Dict[str, Any]:
    """
    Process a PDF file (CV or job posting) and store its embeddings in MongoDB and Qdrant.
    
    Args:
        file_path: Path to the PDF file.
        collection_name: Name of the collection to store embeddings in.
        user_id: ID of the user who owns the documents.
        document_type: Type of document ('cv' or 'job').
        chunk_size: Maximum size of each chunk in tokens.
        chunk_overlap: Number of overlapping tokens between chunks.
        batch_size: Batch size for processing embeddings.
        source: Source of the documents (e.g., 'cv_upload', 'job_posting').
        text_processor: Optional pre-initialized TextProcessor instance.
        storage: Optional pre-initialized QdrantStorage instance.
        
    Returns:
        Dictionary with processing information.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"success": False, "error": f"File {file_path} does not exist."}
    
    allowed_extensions = ['.pdf', '.md']
    if file_path.suffix.lower() not in allowed_extensions:
        return {"success": False, "error": f"File {file_path} is not a supported format (PDF or Markdown)."}
    
    try:
        # Initialize processors and storage if not provided
        if text_processor is None:
            print("Initializing new TextProcessor...")
            text_processor = TextProcessor()
        
        if storage is None or storage.collection_name != collection_name:
            print(f"Initializing new QdrantStorage for collection '{collection_name}'...")
            storage = QdrantStorage(collection_name=collection_name)

        text = ""
        document_metadata = {}

        if file_path.suffix.lower() == '.pdf':
            print(f"Processing PDF: {file_path.name}")
            pdf_processor = PDFProcessor(extract_metadata=True)
            pdf_data = pdf_processor.process_pdf(file_path)
            text = pdf_data.get('text', '').strip()
            document_metadata = pdf_data.get('metadata', {})
            document_metadata['num_pages'] = pdf_data.get('num_pages', 0)
        
        elif file_path.suffix.lower() == '.md':
            print(f"Processing Markdown: {file_path.name}")
            text = file_path.read_text(encoding='utf-8')
            document_metadata['num_pages'] = 1 # Markdown is treated as a single page

        if not text:
            return {"success": False, "error": f"No text could be extracted from {file_path}"}

        # Common metadata
        document_metadata.update({
            'file_name': file_path.name,
            'file_path': str(file_path),
            'document_type': document_type,
            'source': source
        })

        # Process document with intelligent semantic chunking
        # This now returns chunks with embeddings already generated
        processed_chunks = text_processor.process_document(
            file_path,
            document_type=document_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        total_chunks = len(processed_chunks)
        print(f"Document processed into {total_chunks} semantic chunks")
        
        # Log chunking strategy details
        if processed_chunks:
            strategy = processed_chunks[0].get('chunk_strategy', 'unknown')
            print(f"Chunking strategy: {strategy}")
            
            # Count complete sections vs fragmented sections
            complete_sections = sum(1 for chunk in processed_chunks if chunk.get('is_complete_section', False))
            fragmented_sections = total_chunks - complete_sections
            print(f"Complete sections: {complete_sections}, Fragmented chunks: {fragmented_sections}")

        # Prepare chunks for storage (embeddings already included)
        chunks_to_store = []
        for chunk_data in processed_chunks:
            chunk_to_store = {
                "text": chunk_data['text'],
                "embedding": chunk_data['embedding']  # Already in list format
            }
            
            # Add semantic metadata if available
            if 'section' in chunk_data:
                chunk_to_store['section'] = chunk_data['section']
            if 'is_complete_section' in chunk_data:
                chunk_to_store['is_complete_section'] = chunk_data['is_complete_section']
            if 'chunk_index' in chunk_data:
                chunk_to_store['chunk_index'] = chunk_data['chunk_index']
            if 'total_chunks_in_section' in chunk_data:
                chunk_to_store['total_chunks_in_section'] = chunk_data['total_chunks_in_section']
            if 'num_tokens' in chunk_data:
                chunk_to_store['num_tokens'] = chunk_data['num_tokens']
                
            chunks_to_store.append(chunk_to_store)

        # Store all chunks in a single batch operation
        document_id = str(file_path.stem)
        stored_ids = storage.store_embeddings(
            document_id=document_id,
            chunks=chunks_to_store,
            metadata=document_metadata,
            user_id=user_id,
            batch_size=batch_size
        )
        print(f"Stored {len(stored_ids)} chunks for document {document_id}")
        
        return {
            "success": True,
            "total_documents_processed": total_chunks,
            "collection_name": collection_name,
            "storage_backends": ["mongodb", "qdrant"],
            "document_type": document_type,
            "file_name": file_path.name
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error processing file {file_path}: {str(e)}"
        }

def process_pdf_directory(
    directory: Union[str, Path],
    collection_name: str = "cv_embeddings",
    user_id: str = None,
    document_type: str = "cv",
    recursive: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Process all PDF files in a directory.
    
    Args:
        directory: Directory containing PDF files.
        collection_name: Name of the collection to store embeddings in.
        user_id: ID of the user who owns the documents.
        document_type: Type of documents ('cv' or 'job').
        recursive: Whether to process subdirectories.
        **kwargs: Additional arguments to pass to process_pdf_file.
        
    Returns:
        Dictionary with processing summary.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {"success": False, "error": f"Directory {directory} does not exist."}
    
    # Initialize processors and storage if not provided
    text_processor = kwargs.get('text_processor')
    if text_processor is None:
        text_processor = TextProcessor()

    storage = kwargs.get('storage')
    if storage is None or storage.collection_name != collection_name:
        storage = QdrantStorage(collection_name=collection_name)

    # Find all PDF files
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(directory.glob(pattern))
    
    if not pdf_files:
        return {"success": False, "error": f"No PDF files found in {directory}"}
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    results = []
    
    # Process each PDF file
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i}/{len(pdf_files)}: {pdf_file.name}")
        
        result = process_document_file(
            file_path=pdf_file,
            collection_name=collection_name,
            user_id=user_id,
            document_type=document_type,
            source=f"directory_scan_{directory.name}",
            text_processor=text_processor, # Pass instances
            storage=storage, # Pass instances
            **kwargs
        )
        
        results.append({
            'file': str(pdf_file),
            'success': result['success'],
            'chunks_processed': result.get('total_documents_processed', 0) if result['success'] else 0,
            'error': result.get('error')
        })
    
    # Calculate summary statistics
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_chunks = sum(r['chunks_processed'] for r in results if r['success'])
    
    return {
        "success": True,
        "files_processed": len(results),
        "files_successful": successful,
        "files_failed": failed,
        "total_chunks_processed": total_chunks,
        "collection_name": collection_name,
        "document_type": document_type,
        "results": results
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Procesa documentos (CVs o ofertas de trabajo) y almacena sus embeddings.")
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('path', type=str, 
                           help='Path to PDF file or directory containing PDFs')
    input_group.add_argument('--recursive', '-r', action='store_true',
                           help='Process PDFs in subdirectories recursively')
    
    # Document type options
    type_group = parser.add_argument_group('Document Type')
    type_group.add_argument('--document-type', type=str, choices=['cv', 'job'], default='cv',
                          help='Type of documents being processed (default: cv)')
    type_group.add_argument('--source', type=str, default='cv_upload',
                          help='Source identifier (e.g., cv_upload, job_posting)')
    
    # Processing options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument('--collection', type=str, default='cv_embeddings',
                             help='Collection name for storing embeddings (default: cv_embeddings)')
    process_group.add_argument('--user-id', type=str, help='ID del usuario (opcional).')
    process_group.add_argument('--chunk-size', type=int, default=None,
                             help='Maximum tokens per chunk (uses optimized defaults: CV=1500, Job=1000)')
    process_group.add_argument('--overlap', type=int, default=None,
                             help='Overlap between chunks in tokens (uses optimized defaults: CV=300, Job=200)')
    process_group.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for processing embeddings (default: 32)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"Starting document processing for {args.path}")
    print("-" * 70)
    print(f"Document type: {args.document_type.upper()}")
    print(f"Source: {args.source}")
    print(f"Collection: {args.collection}")
    print(f"User ID: {args.user_id or 'Not specified'}")
    # Mostrar configuración de chunking (optimizada o personalizada)
    if args.chunk_size is None:
        chunk_info = f"Optimized for {args.document_type.upper()} (CV=1500, Job=1000 tokens)"
    else:
        chunk_info = f"{args.chunk_size} tokens (custom)"
    
    if args.overlap is None:
        overlap_info = f"Optimized for {args.document_type.upper()} (CV=300, Job=200 tokens)"
    else:
        overlap_info = f"{args.overlap} tokens (custom)"
    
    print(f"Chunk size: {chunk_info}")
    print(f"Chunk overlap: {overlap_info}")
    print(f"Batch size: {args.batch_size}")
    print("="*70 + "\n")
    
    # Initialize processors once for efficiency
    text_processor = TextProcessor()

    # Determine if input is a file or directory
    input_path = Path(args.path)
    
    if input_path.is_file() and input_path.suffix.lower() in ['.pdf', '.md']:
        # Process a single document file (PDF or MD)
        storage = QdrantStorage(collection_name=args.collection)
        result = process_document_file(
            file_path=input_path,
            collection_name=args.collection,
            user_id=args.user_id,
            document_type=args.document_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            batch_size=args.batch_size,
            source=args.source,
            text_processor=text_processor,
            storage=storage
        )
    elif input_path.is_dir():
        # Process a directory of PDFs
        storage = QdrantStorage(collection_name=args.collection)
        result = process_pdf_directory(
            directory=input_path,
            collection_name=args.collection,
            user_id=args.user_id,
            document_type=args.document_type,
            recursive=args.recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            batch_size=args.batch_size,
            source=args.source,
            text_processor=text_processor,
            storage=storage
        )
    else:
        print(f"Error: {input_path} is not a valid PDF file or directory")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("-" * 70)
    
    if result.get('success', False):
        if 'files_processed' in result:  # Directory processing result
            print(f"Files processed: {result['files_processed']}")
            print(f"  ✓ Successful: {result['files_successful']}")
            print(f"  ✗ Failed: {result['files_failed']}")
            print(f"Total chunks processed: {result['total_chunks_processed']}")
        else:  # Single file processing result
            doc_id = Path(result.get('file_name', '')).stem
            print(f"File: {result.get('file_name', 'Unknown')}")
            print(f"Chunks processed: {result.get('total_documents_processed', 0)}")
            # Imprimir el ID del documento para que el pipeline lo capture
            if doc_id:
                print(f"PROCESSED_DOC_ID:{doc_id}")
        
        print(f"\nCollection: {result['collection_name']}")
        print(f"Storage backends: {', '.join(result.get('storage_backends', ['mongodb', 'qdrant']))}")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error occurred')}")
    
    print("="*70 + "\n")
    
    if not result.get('success', False):
        sys.exit(1)

if __name__ == "__main__":
    main()
