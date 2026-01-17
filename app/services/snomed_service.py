# app/services/snomed_service.py
import requests

# URL base de Snowstorm
SNOMED_URL = "http://10.10.0.37:8080/browser/MAIN/concepts"

def buscar_conceptos_snomed(term: str, ecl: str = "< 404684003>", limit: int = 10):
    print(f"🔍 BUSCANDO EN SNOMED → Term: '{term}' | ECL: '{ecl}' | Limit: {limit}")
    params = {
        "term": term,
        "ecl": ecl,
        "activeFilter": "true",
        "language": "es",
        "limit": limit
    }
    
    try:
        response = requests.get(SNOMED_URL, params=params)
        response.raise_for_status()  # lanza excepción si falla
        data = response.json()
        
        conceptos = []
        for item in data.get("items", []):
            conceptos.append({
                "conceptId": item.get("conceptId"),
                "preferredTerm": item.get("preferredTerm"),
                "fsn": item.get("fsn", "")
            })
            
        print(f"✅ SNOMED → encontrados {len(conceptos)} conceptos para '{term}'")
        
        
        return conceptos
    
    except Exception as e:
        print(f"[ERROR] Consulta SNOMED fallida: {e}")
        return []

def obtener_sinonimos(concept_id: str, language: str = "es"):
    """
    Devuelve sinónimos activos (no FSN) del concepto en el idioma indicado.
    Usa el endpoint de descriptions del browser.
    """
    try:
        url = f"{SNOMED_URL.rstrip('/concepts')}/concepts/{concept_id}/descriptions"
        params = {"active": "true", "language": language}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json() or {}
        out = []
        for d in data.get("items", []) or []:
            term = d.get("term", "")
            type_id = str(d.get("type", {}).get("conceptId", "")) or str(d.get("typeId", ""))
            # 900000000000013009 = Synonym (en muchas releases)
            if term and "fsn" not in (d.get("acceptabilityMap", {}) or {}) and "Fully specified" not in (d.get("type", {}).get("pt", {}).get("term", "")).lower():
                out.append(term)
        return out
    except Exception:
        return []
