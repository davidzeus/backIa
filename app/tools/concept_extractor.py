# app/tools/concept_extractor.py

from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM

PROMPT_EXTRAER_CONCEPTOS = """
Sos un asistente médico experto en procesamiento de lenguaje clínico.

Tu tarea es analizar el siguiente texto de historia clínica, y EXTRAER UNA LISTA de conceptos clínicos relevantes que deban ser codificados en SNOMED CT.

Reglas:
- No incluyas frases completas, solo conceptos.
- No expliques nada.
- No inventes conceptos.
- Cada concepto debe estar en español, en su forma más precisa y breve.

Ejemplo de salida:

- Hipertensión arterial
- Diabetes mellitus tipo 2
- Colecistectomía
- Dolor torácico

Ahora procesá este texto:

\"\"\"{texto_seccion}\"\"\"

Lista de conceptos clínicos:
"""


def extraer_conceptos_con_llm(llm: LLM, texto_seccion: str) -> list:
    # Preparar prompt
    prompt = PROMPT_EXTRAER_CONCEPTOS.format(texto_seccion=texto_seccion)

    # Llamar al LLM
    response: Response = llm.complete(prompt)
    text_response = response.text.strip()

    # Parsear lista de conceptos
    conceptos = []
    for line in text_response.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            concepto = line[1:].strip()
            if concepto:
                conceptos.append(concepto)

    return conceptos
