import PyPDF2
import docx
from pathlib import Path

def extract_text_from_file(file_path: str) -> str:
    """Extract text from uploaded file based on extension."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    try:
        if extension == '.pdf':
            return extract_pdf_text(file_path)
        elif extension == '.txt':
            return extract_txt_text(file_path)
        elif extension == '.docx':
            return extract_docx_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_txt_text(file_path: Path) -> str:
    """Extract text from TXT file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_docx_text(file_path: Path) -> str:
    """Extract text from DOCX file."""
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def save_uploaded_file(uploaded_file, upload_dir: Path) -> str:
    """Save uploaded file and return path."""
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)
