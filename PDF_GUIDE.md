# Using Enterprise PDFs with RAG Evaluation

## Quick Setup

### 1. Ensure pypdf is installed
```powershell
pip install pypdf
```
✅ Already installed in your environment!

## 2. Add Your PDFs to the docs/ Folder

```powershell
# Copy your PDFs
cp path/to/your/enterprise_doc.pdf docs/
cp path/to/another_document.pdf docs/

# Or move them
mv path/to/pdfs/*.pdf docs/
```

## 3. Run the Evaluation

```powershell
# Standard evaluation (now includes PDFs)
.venv\Scripts\python -m rag_eval.cli --docs docs --queries queries.json

# View results
.venv\Scripts\python show_results.py
```

That's it! The system now automatically:
- ✅ Detects PDF files
- ✅ Extracts text from all pages
- ✅ Processes them alongside .md and .txt files
- ✅ Shows page count during loading

---

## Supported File Formats

| Format | Extension | Status |
|--------|-----------|--------|
| PDF | `.pdf` | ✅ Supported |
| Markdown | `.md` | ✅ Supported |
| Text | `.txt` | ✅ Supported |

---

## Example Output When Loading PDFs

```
Loaded: sample.md
Loaded: enterprise_report.pdf (45 pages)
Loaded: technical_spec.pdf (128 pages)
Loaded: user_guide.txt

Evaluating 3 documents across 4 strategies...
```

---

## Tips for Enterprise PDFs

### 1. **Scanned PDFs (Images)**
If your PDFs are scanned documents:
```powershell
# Install OCR support
pip install pytesseract pillow pdf2image

# You'll need Tesseract OCR installed
# Windows: choco install tesseract
# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. **Password-Protected PDFs**
```python
# Unlock PDFs before processing
from pypdf import PdfReader

reader = PdfReader("protected.pdf", password="your_password")
```

### 3. **Large PDF Collections**
```powershell
# Process PDFs in batches
python -m rag_eval.cli --docs docs/batch1
python -m rag_eval.cli --docs docs/batch2 --output results_batch2.json
```

### 4. **Extract Metadata**
PDFs contain useful metadata:
```python
from pypdf import PdfReader

reader = PdfReader("enterprise_doc.pdf")
metadata = reader.metadata
print(f"Title: {metadata.title}")
print(f"Author: {metadata.author}")
print(f"Pages: {len(reader.pages)}")
```

---

## Custom PDF Processing

If you need advanced PDF processing, create a custom loader:

### Option 1: Pre-process PDFs

```python
# preprocess_pdfs.py
from pypdf import PdfReader
import os

def extract_pdf_to_txt(pdf_path, output_dir="docs"):
    reader = PdfReader(pdf_path)
    text_parts = []
    
    for page in reader.pages:
        text_parts.append(page.extract_text())
    
    full_text = "\n\n".join(text_parts)
    
    # Save as .txt
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"Converted: {pdf_path} → {txt_path}")

# Usage
extract_pdf_to_txt("enterprise_report.pdf")
```

### Option 2: Use LlamaIndex PDF Readers

```python
# Install LlamaIndex readers
pip install llama-index-readers-file

# Use in your code
from llama_index.readers.file import PDFReader

reader = PDFReader()
documents = reader.load_data(file="enterprise_doc.pdf")
text = "\n\n".join([doc.text for doc in documents])
```

---

## Handling Common Issues

### Issue: "No extractable text"
**Cause:** PDF contains images/scans, not text
**Solution:** Use OCR (pytesseract) or convert to text-based PDF

### Issue: Garbled characters
**Cause:** Font encoding issues
**Solution:** Try different extraction methods:
```python
# Method 1: Standard
text = page.extract_text()

# Method 2: Layout mode
text = page.extract_text(extraction_mode="layout")

# Method 3: Clean text
import re
text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
```

### Issue: Tables not extracted properly
**Cause:** Tables are complex structures
**Solution:** Use specialized libraries:
```python
pip install tabula-py  # For tables
pip install pdfplumber  # Better table extraction

import pdfplumber
with pdfplumber.open("doc.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
```

---

## Directory Structure Example

```
Chunk-RAG-Python/
├── docs/
│   ├── enterprise_report_2024.pdf      ← Your PDFs here
│   ├── technical_specification.pdf
│   ├── user_manual.pdf
│   ├── policy_document.md
│   └── readme.txt
├── queries.json                         ← Your evaluation queries
└── evaluation_results.json              ← Generated results
```

---

## Example Workflow

```powershell
# 1. Add your enterprise PDFs
cp C:\Enterprise\Reports\*.pdf docs\

# 2. Create relevant queries
# Edit queries.json with questions about your PDFs

# 3. Run evaluation
.venv\Scripts\python -m rag_eval.cli

# 4. Analyze results
.venv\Scripts\python show_results.py

# 5. Review detailed results
code evaluation_results.json
```

---

## Performance Considerations

### PDF Size Optimization

| PDF Pages | Processing Time | Recommendation |
|-----------|----------------|----------------|
| 1-50 | < 1 second | Process directly |
| 51-200 | 1-5 seconds | Process directly |
| 201-500 | 5-15 seconds | Consider batching |
| 500+ | 15+ seconds | Split into multiple files |

### Memory Usage

Large PDFs (500+ pages) may use significant memory:
```python
# Process PDFs in chunks if needed
def process_large_pdf(pdf_path, pages_per_chunk=100):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)
        chunk_text = []
        
        for i in range(start, end):
            chunk_text.append(reader.pages[i].extract_text())
        
        # Process this chunk
        yield "\n\n".join(chunk_text)
```

---

## Advanced: Maintaining PDF Structure

If your PDFs have important structure (headers, sections):

```python
from pypdf import PdfReader

def extract_with_structure(pdf_path):
    reader = PdfReader(pdf_path)
    structured_text = []
    
    for i, page in enumerate(reader.pages):
        # Add page markers
        structured_text.append(f"\n\n--- Page {i+1} ---\n\n")
        
        # Extract text with layout
        text = page.extract_text(extraction_mode="layout")
        structured_text.append(text)
    
    return "".join(structured_text)
```

This helps StructureChunker maintain document hierarchy.

---

## Next Steps

1. **Add your PDFs** to `docs/` folder
2. **Update queries.json** with questions specific to your documents
3. **Run evaluation** to see which chunking strategy works best for your enterprise content
4. **Analyze results** to optimize your RAG system

---

## Questions?

- PDFs not loading? Check file permissions and encoding
- Need OCR? Install pytesseract + Tesseract
- Complex tables? Try pdfplumber or tabula-py
- Very large PDFs? Consider splitting or batch processing

Your system is now ready to evaluate enterprise PDFs! 🚀
