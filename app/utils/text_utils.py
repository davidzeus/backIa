import re
import unicodedata


def normalizar_texto(texto: str) -> str:
    """
    Limpia y normaliza texto clínico para asegurar máxima consistencia antes del procesamiento.

    Pipeline de limpieza:
    1. Normaliza Unicode a NFKC para unificar caracteres con apariencias similares.
    2. Elimina etiquetas estilo HTML/XML como <DESCRIPCION>, etc.
    3. Reemplaza cualquier secuencia de caracteres de espaciado (saltos de línea, tabs, etc.) por un solo espacio.
    4. Elimina espacios en blanco al principio y al final del texto.
    """
    if not texto:
        return ""

    # 1. Normalización Unicode (crucial para texto de PDFs, Word, etc.)
    texto_normalizado = unicodedata.normalize("NFKC", texto)

    # 2. Elimina etiquetas HTML/XML (tu lógica original, es perfecta para esto)
    texto_sin_etiquetas = re.sub(r"<[^>]+>", "", texto_normalizado)

    # 3. Reemplaza múltiples espacios, tabs o saltos de línea por un solo espacio (más eficiente)
    texto_unificado = re.sub(r"\s+", " ", texto_sin_etiquetas)

    # 4. Limpia espacios al principio y final
    return texto_unificado.strip()
