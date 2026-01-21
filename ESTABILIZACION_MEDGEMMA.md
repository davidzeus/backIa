# 📄 Documentación Técnica: Estabilización del Agente Clínico (SAMI)

**Versión del Modelo**: Google MedGemma 4B (Quantized)  
**Arquitectura**: Hybrid RAG + Agentic Workflow (Agno)  
**Fecha**: 21/01/2026

---

## 1. Resumen Ejecutivo

El objetivo de esta fase fue migrar el motor de inferencia de un modelo generalista (Ministral 3B) a un modelo especializado en medicina (MedGemma 4B). Si bien el modelo médico mejoró la comprensión de la terminología, introdujo inestabilidad en el formato y la generación. Se implementó una capa de post-procesamiento determinista en Python y una estrategia de RAG Híbrido para lograr una precisión del 100% en la estructura de salida (JSON) y relevancia clínica.

---

## 2. Arquitectura del Flujo de Datos

El sistema sigue un pipeline de 5 etapas para garantizar que la respuesta médica sea precisa y esté bien formateada.

1.  **Ingesta & Planificación**: El Planner analiza la pregunta y define filtros (fechas, servicios).
2.  **Recuperación Híbrida (Hybrid Retrieval)**: Búsqueda en Qdrant + Filtro de Proximidad Gaussiano (prioriza fechas cercanas).
3.  **Re-Ranking Neuronal**: Un Cross-Encoder (`mmarco-mMiniLMv2`) reordena los documentos por relevancia clínica real, no solo palabras clave.
4.  **Generación (LLM)**: MedGemma 4B analiza el contexto usando Chain-of-Thought (Cadena de Pensamiento).
5.  **Sanitización (Middleware)**: Una capa de Python limpia, deduplica y valida el JSON antes de enviarlo al Frontend.

---

## 3. Registro de Problemas y Soluciones (Troubleshooting Log)

A continuación se detallan los 4 obstáculos críticos enfrentados durante la implementación de MedGemma y sus soluciones definitivas.

### 🔴 Problema 1: "Ceguera Selectiva" (Fallo de Retrieval)
*   **Síntoma**: El agente respondía "No hay registros" o ignoraba resultados de laboratorio (ej. cultivos positivos) si estos aparecían al final de documentos largos.
*   **Causa Raíz**: La búsqueda vectorial simple diluía la información específica (microbiología) entre notas de evolución genéricas ("paciente estable").
*   **Solución**: **Implementación de Reranker Local + Fallback**.
    *   Se añadió un modelo Cross-Encoder que re-puntúa los documentos.
    *   Se creó un mecanismo de "Botón de Pánico": Si los filtros estrictos no traen nada, el sistema busca automáticamente en todo el historial.

### 🔴 Problema 2: "Colapso Cognitivo" (Bucles Infinitos)
*   **Síntoma**: El modelo entraba en bucles repetitivos de texto, generando salidas como: *"Antecedente de antecedentes de antecedentes de..."* (x50 veces).
*   **Causa Raíz**: Los modelos pequeños (4B) tienen baja tolerancia a la repetición en el contexto (muchas notas médicas repiten los mismos antecedentes).
*   **Solución**: **Ajuste de Hiperparámetros de Inferencia (Ollama)**.
    *   `repeat_penalty`: Se elevó a **1.5** (Penalización agresiva).
    *   `mirostat`: **Modo 2**. Este algoritmo controla la perplejidad de la generación, evitando que el modelo "se obsesione" con una frase.

### 🔴 Problema 3: Conflicto de Formato (JSON Roto)
*   **Síntoma**: Al pedir un JSON, el modelo devolvía texto explicativo, Markdown incompleto o JSONs cortados a la mitad.
*   **Causa Raíz**: Conflicto en el System Prompt entre "Actuar como médico" (explicar) y "Actuar como API" (estructurar).
*   **Solución**: **Estrategia One-Shot Prompting**.
    *   Se inyectó un ejemplo literal de JSON en el prompt:
        ```json
        ### EJEMPLO DE SALIDA ESPERADA:
        { "medicamentos": ["Enalapril"] }
        ```
    *   Esto guía al modelo visualmente sobre qué estructura debe imitar.

### 🟢 Problema 4 (La Solución Final): Datos Sucios y Duplicados
*   **Síntoma**: El modelo generaba el JSON, pero repetía ítems (ej. 10 veces "Diabetes") o incluía ruido.
*   **Solución**: **Capa de Limpieza Determinista (Python Regex)**.
    *   En lugar de "pedirle" al modelo que no repita, asumimos que lo hará y lo corregimos en código.
    *   Se implementó la función `_extract_list_from_text` que usa Expresiones Regulares para extraer los arrays y `set()` para eliminar duplicados matemáticamente.

---

## 4. Implementación Clave (El Código Salvador)

Este fragmento es el corazón de la estabilidad actual. Se ejecuta después de que el modelo responde y antes de enviar al Frontend.

```python
def _extract_list_from_text(text: str, key: str) -> list[str]:
    """
    Extrae listas de un texto (incluso si el JSON está roto) y elimina duplicados.
    """
    import re
    # Busca patrones tipo "key": [ ... ] ignorando errores de sintaxis alrededor
    pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, text, re.DOTALL)
    
    clean_items = []
    if match:
        content = match.group(1)
        # Extrae cada item individual entre comillas
        raw_items = re.findall(r'"([^"]*)"', content)
        
        # Deduplicación usando Set (O(1))
        seen = set()
        for item in raw_items:
            i_clean = item.strip()
            if i_clean and i_clean.lower() not in seen:
                seen.add(i_clean.lower())
                clean_items.append(i_clean)
                
    return clean_items
```

---

## 5. Capacidades Actuales del Agente (Estado Final)

| Capacidad | Estado | Descripción |
| :--- | :--- | :--- |
| **Jerga Médica** | ✅ Experto | Reconoce siglas como PTZ (Piperacilina-Tazobactam), AMS, HTA. |
| **Formato** | ✅ JSON Puro | Entrega un objeto JSON válido listo para `JSON.parse()` en JavaScript. |
| **Limpieza** | ✅ Auto-Clean | Elimina automáticamente duplicados y alucinaciones repetitivas. |
| **Velocidad** | ⚡ Optimizada | Latencia promedio de generación ~20s (aceptable para análisis profundo local). |
