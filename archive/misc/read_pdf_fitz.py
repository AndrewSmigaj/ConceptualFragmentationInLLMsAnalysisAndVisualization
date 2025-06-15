#!/usr/bin/env python3

import os
import sys

try:
    import fitz  # PyMuPDF
    use_fitz = True
except ImportError:
    use_fitz = False

try:
    import PyPDF2
    use_pypdf2 = True
except ImportError:
    use_pypdf2 = False

def read_with_fitz(pdf_path):
    """Read PDF using PyMuPDF (fitz)"""
    doc = fitz.open(pdf_path)
    print(f"Document: {pdf_path}")
    print(f"Pages: {doc.page_count}")
    print("=" * 80)
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        print(f"\n--- Page {page_num + 1} ---\n")
        text = page.get_text()
        print(text)
        print("=" * 80)
    
    doc.close()

def read_with_pypdf2(pdf_path):
    """Read PDF using PyPDF2"""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        print(f"Document: {pdf_path}")
        print(f"Pages: {len(pdf_reader.pages)}")
        print("=" * 80)
        
        for page_num, page in enumerate(pdf_reader.pages):
            print(f"\n--- Page {page_num + 1} ---\n")
            text = page.extract_text()
            print(text)
            print("=" * 80)

def main():
    pdf_path = "arxiv_submission/main.pdf"
    output_file = "arxiv_submission/main_text.txt"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    print(f"Available libraries: fitz={use_fitz}, PyPDF2={use_pypdf2}")
    
    # Redirect stdout to handle encoding issues
    original_stdout = sys.stdout
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            if use_fitz:
                print("Using PyMuPDF (fitz)...")
                try:
                    read_with_fitz(pdf_path)
                except Exception as e:
                    print(f"Error with fitz: {e}")
                    if use_pypdf2:
                        print("\nFalling back to PyPDF2...")
                        read_with_pypdf2(pdf_path)
            elif use_pypdf2:
                print("Using PyPDF2...")
                read_with_pypdf2(pdf_path)
            else:
                print("No PDF libraries available. Please install PyMuPDF or PyPDF2:")
                print("  pip install PyMuPDF  # or")
                print("  pip install PyPDF2")
    finally:
        sys.stdout = original_stdout
        print(f"PDF text extracted to: {output_file}")

if __name__ == "__main__":
    main()