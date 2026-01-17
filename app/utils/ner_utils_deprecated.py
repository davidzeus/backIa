# app/utils/ner_utils.py
'''
from transformers import pipeline

# Cargamos el pipeline NER
ner_pipeline = pipeline("ner", model="PlanTL-GOB-ES/roberta-base-bne-ner-health", aggregation_strategy="simple")

def extract_entities_with_ner(clinical_text: str):
    """
    Extrae entidades clínicas usando modelo NER especializado en español.
    """
    ner_results = ner_pipeline(clinical_text)

    entities = []
    for entity in ner_results:
        entities.append({
            "text": entity["word"],
            "type": entity["entity_group"],
            "score": entity["score"]
        })

    return entities
'''
