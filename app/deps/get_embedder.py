from app.infra.ml_providers import get_embedder

def embed_query_project(text: str) -> list[float]:
    emb = get_embedder()
    if hasattr(emb, "embed_query"):
        vec = emb.embed_query(text)
    elif hasattr(emb, "encode"):
        out = emb.encode([text])
        vec = out[0]
    else:
        raise RuntimeError("El embedder no expone embed_query/encode")
    return vec.tolist() if hasattr(vec, "tolist") else [float(x) for x in vec]
