import os
import pymupdf  # PyMuPDF
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime

# Import time utilities
from src.utils.time_utils import get_current_utc_timestamp, to_iso_format, to_unix_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """A utility class for processing PDF files, extracting text and metadata."""
    
    def __init__(self, extract_metadata: bool = True):
        """
        Initialize the PDF processor.
        
        Args:
            extract_metadata: Whether to extract metadata from PDF files
        """
        self.extract_metadata = extract_metadata
    
    def process_pdf(self, file_path: Union[str, Path, bytes]) -> Dict[str, any]:
        """
        Process a PDF file and extract its text content and metadata.
        
        Args:
            file_path: Path to the PDF file or binary PDF data
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Extracted metadata (if extract_metadata is True)
                - num_pages: Number of pages in the PDF
        """
        try:
            # Handle binary data
            if isinstance(file_path, bytes):
                import io
                doc = pymupdf.open(stream=io.BytesIO(file_path), filetype='pdf')
            else:
                # Handle file path
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                doc = pymupdf.open(file_path)
            
            with doc:
                # Extract text from all pages
                text = ""
                for page in doc:
                    text += page.get_text() + "\n\n"
                
                # Extract metadata if requested
                metadata = {}
                if self.extract_metadata and doc.metadata:
                    metadata = self._clean_metadata(doc.metadata)
                
                # Add processing timestamp
                metadata['processed_at'] = get_current_utc_timestamp()
                
                return {
                    'text': text.strip(),
                    'metadata': metadata,
                    'num_pages': len(doc)
                }
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def _clean_metadata(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """
        Clean and format PDF metadata.
        
        Args:
            metadata: Raw PDF metadata
            
        Returns:
            Cleaned metadata dictionary with normalized timestamps
        """
        cleaned = {}
        timestamp_fields = ['creationdate', 'moddate', 'processed_at']
        
        for key, value in metadata.items():
            if not value:  # Skip None or empty values
                continue
                
            # Convert key to lowercase and replace spaces with underscores
            clean_key = key.lower().replace(' ', '_')
            
            # Handle timestamp fields
            if clean_key in timestamp_fields and value:
                try:
                    # Convertir a formato Unix timestamp si es posible
                    if isinstance(value, (int, float)):
                        cleaned[clean_key] = int(value)
                    elif isinstance(value, str):
                        # Intentar parsear la fecha del PDF
                        if value.startswith('D:'):
                            # Formato común en PDFs: D:YYYYMMDDHHmmSSOHH'mm'
                            date_str = value[2:18]  # Tomar solo la parte de la fecha
                            dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                            cleaned[clean_key] = int(dt.timestamp())
                        else:
                            # Intentar con otros formatos
                            cleaned[clean_key] = to_unix_timestamp(value) or value
                    elif hasattr(value, 'timestamp'):
                        # Si es un objeto datetime
                        cleaned[clean_key] = int(value.timestamp())
                    else:
                        cleaned[clean_key] = value
                except Exception as e:
                    logger.warning(f"Error al procesar campo de fecha {clean_key}: {e}")
                    cleaned[clean_key] = value
            else:
                cleaned[clean_key] = value
                
        return cleaned
    
    def process_directory(self, directory: Union[str, Path], 
                         recursive: bool = False) -> List[Dict[str, any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory: Directory containing PDF files
            recursive: Whether to process subdirectories
            
        Returns:
            List of processed PDF documents
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        results = []
        
        # Define the pattern for PDF files
        pattern = "**/*.pdf" if recursive else "*.pdf"
        
        # Process each PDF file
        for pdf_file in directory.glob(pattern):
            if pdf_file.is_file():
                try:
                    result = self.process_pdf(pdf_file)
                    result["file_path"] = str(pdf_file)
                    result["file_name"] = pdf_file.name
                    results.append(result)
                    logger.info(f"Processed PDF: {pdf_file.name} ({result['num_pages']} pages)")
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}: {str(e)}")
        
        return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF files and extract text with metadata.")
    parser.add_argument("input", help="PDF file or directory containing PDFs")
    parser.add_argument("--recursive", "-r", action="store_true", 
                       help="Process PDFs in subdirectories recursively")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    processor = PDFProcessor()
    
    input_path = Path(args.input)
    results = []
    
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # Process a single PDF file
        try:
            result = processor.process_pdf(input_path)
            result["file_path"] = str(input_path)
            result["file_name"] = input_path.name
            results.append(result)
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
    elif input_path.is_dir():
        # Process a directory of PDFs
        results = processor.process_directory(input_path, recursive=args.recursive)
    else:
        print(f"Error: {input_path} is not a valid PDF file or directory")
        sys.exit(1)
    
    # Print or save results
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        for result in results:
            print("\n" + "="*50)
            print(f"File: {result['file_name']}")
            print(f"Pages: {result['num_pages']}")
            if result['metadata']:
                print("\nMetadata:")
                for key, value in result['metadata'].items():
                    print(f"  {key}: {value}")
            print("\nFirst 500 characters of text:")
            print(result['text'][:500] + ("..." if len(result['text']) > 500 else ""))
