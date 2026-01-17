
import unittest
from unittest.mock import MagicMock, patch
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Mock de dependencias antes de importar rag_tool
with patch('app.helpers.query_helpers._qdrant', MagicMock()), \
     patch('app.infra.ml_providers.get_embedder', MagicMock()):
    from app.tools.rag_tool import search_clinical_history

class TestRagFeedbackLoop(unittest.TestCase):
    
    @patch('app.tools.rag_tool.get_patient_context')
    @patch('app.tools.rag_tool.SafeQdrantVectorStore')
    @patch('app.tools.rag_tool.VectorStoreIndex')
    @patch('app.tools.rag_tool.SentenceTransformerRerank')
    def test_strict_mode_filter(self, mock_rerank, mock_index, mock_store, mock_ctx):
        # Setup
        mock_ctx.return_value = "12345"
        
        # --- CASO 1: Strict=True (Plan A) ---
        filters_strict = {
            "name_hint": "EVOLUCIÓN", 
            "strict": True
        }
        
        # Ejecutamos (el mock del store capturará el filtro)
        search_clinical_history("test query", qdrant_filters=filters_strict)
        
        # Verificamos argumentos de store
        init_args = mock_index.from_vector_store.call_args
        # El retriever se crea DESPUES.
        # En rag_tool: index = VectorStoreIndex.from_vector_store(...) -> Returns instance
        # luego index.as_retriever(...)
        
        index_instance = mock_index.from_vector_store.return_value
        retriever_mock = index_instance.as_retriever
        
        if not retriever_mock.call_args:
             print("DEBUG: as_retriever was NOT called. Calls to index:", mock_index.mock_calls)
             self.fail("as_retriever was not called")
             
        call_kwargs = retriever_mock.call_args[1]
        
        actual_filter = call_kwargs['vector_store_kwargs']['qdrant_filters']
        
        # Debe tener MUST conteniendo el name hint
        # Nota: rag_tool crea un Filter(should=[...]) dentro de MUST para groups.
        print("\n[STRICT] Filter Must:", actual_filter.must)
        print("[STRICT] Filter Should:", actual_filter.should)
        
        # Verificación simple: ¿Hay algo en MUST relacionado con el hint?
        # Revisamos la estructura interna: Filter(should=[FieldCondition(key='name', match=MatchValue(value='EVOLUCIÓN'))])
        # Buscamos manualmente
        found_strict = False
        for cond in actual_filter.must:
            if isinstance(cond, Filter) and cond.should:
                for sub in cond.should:
                    if sub.key == 'name' and sub.match.value == 'EVOLUCIÓN':
                        found_strict = True
        
        self.assertTrue(found_strict, "En STRICT mode, 'name_hint' debe estar dentro de MUST")

        # --- CASO 2: Strict=False (Plan B/Relaxed) ---
        filters_relaxed = {
            "name_hint": "EVOLUCIÓN", 
            "strict": False
        }
        
        search_clinical_history("test query 2", qdrant_filters=filters_relaxed)
        
        # Verificamos segunda llamada
        call_kwargs_2 = retriever_mock.call_args_list[-1][1]
        actual_filter_2 = call_kwargs_2['vector_store_kwargs']['qdrant_filters']
        
        print("\n[RELAXED] Filter Must:", actual_filter_2.must)
        print("[RELAXED] Filter Should:", actual_filter_2.should)
        
        # Debe estar en SHOULD, no en MUST
        found_relaxed_in_should = False
        for cond in actual_filter_2.should:
            # FieldCondition directo
             if hasattr(cond, 'key') and cond.key == 'name' and cond.match.value == 'EVOLUCIÓN':
                 found_relaxed_in_should = True
                 
        self.assertTrue(found_relaxed_in_should, "En RELAXED mode, 'name_hint' debe estar en SHOULD")
        
        # Asegurar que NO está en MUST (salvo el filtro de paciente)
        # Filtro de paciente es: FieldCondition(key='paciente_id', ...)
        # Checkeamos que no haya filtro de name en MUST
        for cond in actual_filter_2.must:
             if isinstance(cond, Filter) and cond.should: # El wrapper de strict
                  self.fail("En RELAXED mode, no debería haber filtros anidados en MUST para name")

if __name__ == '__main__':
    unittest.main()
