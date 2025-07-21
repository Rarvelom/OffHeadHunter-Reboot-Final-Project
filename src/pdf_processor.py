import os
import pymupdf  # PyMuPDF
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging

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
    
    def process_pdf(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process a PDF file and extract its text content and metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Extracted metadata (if extract_metadata is True)
                - num_pages: Number of pages in the PDF
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            with pymupdf.open(file_path) as doc:
                # Extract text from all pages
                text = ""
                for page in doc:
                    text += page.get_text() + "\n\n"
                
                # Extract metadata if requested
                metadata = {}
                if self.extract_metadata and doc.metadata:
                    metadata = self._clean_metadata(doc.metadata)
                
                return {
                    "text": text.strip(),
                    "metadata": metadata,
                    "num_pages": len(doc)
                }
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def _clean_metadata(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """
        Clean and format PDF metadata.
        
        Args:
            metadata: Raw PDF metadata
            
        Returns:
            Cleaned metadata dictionary
        """
        cleaned = {}
        for key, value in metadata.items():
            if value:  # Skip None or empty values
                # Convert key to lowercase and replace spaces with underscores
                clean_key = key.lower().replace(' ', '_')
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
