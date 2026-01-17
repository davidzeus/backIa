# app/config/logging_setup.py
import logging
import os

NOISY_LIBS = (
    "httpx", "httpcore", "urllib3", "requests",
    "huggingface_hub", "transformers", "sentence_transformers",
    "qdrant_client", "llama_index", "watchfiles",
    "uvicorn", "uvicorn.error", "uvicorn.access"
)

def setup_logging() -> None:
    """
    - Por defecto: root=ERROR (no se ve casi nada)
    - Si HCI_DEBUG=1: root=DEBUG y mostramos logs de nuestra app
    - Access log de Uvicorn apagado salvo que HCI_DEBUG=1
    """
    hci_debug = os.getenv("HCI_DEBUG", "0") == "1"

    # Nivel raíz
    root_level = logging.DEBUG if hci_debug else logging.ERROR

    logging.basicConfig(
        level=root_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,  # pisa cualquier config previa (incluida la de uvicorn)
    )

    # Bajar volumen de libs ruidosas SIEMPRE
    for name in NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Nuestra app: visible solo en modo debug
    app_level = logging.DEBUG if hci_debug else logging.ERROR
    logging.getLogger("app").setLevel(app_level)
    logging.getLogger("app.services").setLevel(app_level)

    # Uvicorn access log: siempre apagado salvo debug
    logging.getLogger("uvicorn.access").disabled = not hci_debug

    # “Application startup complete.” viene de uvicorn.error (INFO).
    # Lo bajamos a ERROR cuando no estamos en debug.
    if not hci_debug:
        logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
