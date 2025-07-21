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
from src.unified_storage import UnifiedStorage
from src.pdf_processor import PDFProcessor

# Load environment variables
load_dotenv()

def process_pdf_file(
    file_path: Union[str, Path],
    collection_name: str = "cv_embeddings",
    user_id: str = None,
    document_type: str = "cv",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32,
    source: str = "cv_upload"
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
        
    Returns:
        Dictionary with processing information.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"success": False, "error": f"File {file_path} does not exist."}
    
    if file_path.suffix.lower() != '.pdf':
        return {"success": False, "error": f"File {file_path} is not a PDF."}
    
    try:
        # Initialize text processor, PDF processor, and storage
        text_processor = TextProcessor()
        pdf_processor = PDFProcessor(extract_metadata=True)
        storage = UnifiedStorage(
            collection_name=collection_name,
            user_id=user_id
        )
        
        # Process the PDF file
        print(f"Processing PDF: {file_path.name}")
        pdf_data = pdf_processor.process_pdf(file_path)
        
        # Extract text and metadata
        text = pdf_data.get('text', '').strip()
        if not text:
            return {"success": False, "error": f"No text could be extracted from {file_path}"}
        
        # Get metadata from PDF or use defaults
        metadata = pdf_data.get('metadata', {})
        metadata.update({
            'file_name': file_path.name,
            'file_path': str(file_path),
            'document_type': document_type,
            'num_pages': pdf_data.get('num_pages', 0),
            'user_id': user_id
        })
        
        # Split text into chunks
        chunks = text_processor.split_into_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        total_chunks = len(chunks)
        print(f"Document split into {total_chunks} chunks")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Generate a unique ID for the chunk
            document_id = f"{file_path.stem}-chunk-{i}"
            
            # Generate embedding for the chunk
            embedding = text_processor.embed_text(chunk)
            
            # Add chunk-specific metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': total_chunks,
                'chunk_size': len(chunk.split())  # Approximate word count
            })
            
            # Store in MongoDB and Qdrant
            storage.store_embedding(
                text=chunk,
                embedding=embedding,
                document_id=document_id,
                metadata=chunk_metadata,
                source=source
            )
            
            if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
                print(f"Processed chunk {i + 1}/{total_chunks}")
        
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
        
        result = process_pdf_file(
            file_path=pdf_file,
            collection_name=collection_name,
            user_id=user_id,
            document_type=document_type,
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
    parser = argparse.ArgumentParser(description='Process PDF documents and generate embeddings.')
    
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
    process_group.add_argument('--user-id', type=str, default=None,
                             help='ID of the user who owns the documents')
    process_group.add_argument('--chunk-size', type=int, default=1000,
                             help='Maximum tokens per chunk (default: 1000)')
    process_group.add_argument('--overlap', type=int, default=200,
                             help='Overlap between chunks in tokens (default: 200)')
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
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Chunk overlap: {args.overlap} tokens")
    print(f"Batch size: {args.batch_size}")
    print("="*70 + "\n")
    
    # Determine if input is a file or directory
    input_path = Path(args.path)
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # Process a single PDF file
        result = process_pdf_file(
            file_path=input_path,
            collection_name=args.collection,
            user_id=args.user_id,
            document_type=args.document_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            batch_size=args.batch_size,
            source=args.source
        )
    elif input_path.is_dir():
        # Process a directory of PDFs
        result = process_pdf_directory(
            directory=input_path,
            collection_name=args.collection,
            user_id=args.user_id,
            document_type=args.document_type,
            recursive=args.recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            batch_size=args.batch_size,
            source=args.source
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
            print(f"File: {result.get('file_name', 'Unknown')}")
            print(f"Chunks processed: {result.get('total_documents_processed', 0)}")
        
        print(f"\nCollection: {result['collection_name']}")
        print(f"Storage backends: {', '.join(result.get('storage_backends', ['mongodb', 'qdrant']))}")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error occurred')}")
    
    print("="*70 + "\n")
    
    if not result.get('success', False):
        sys.exit(1)

if __name__ == "__main__":
    main()
