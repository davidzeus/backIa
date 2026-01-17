from docx import Document as DocxDocument


def leer_docx(ruta_archivo: str) -> str:
    """
    Lee un archivo .docx y devuelve el texto concatenado en un solo string.
    """
    try:
        doc = DocxDocument(ruta_archivo)
        texto = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return texto.strip()
    except Exception as e:
        raise RuntimeError(f"Error al leer el archivo DOCX: {e}")
