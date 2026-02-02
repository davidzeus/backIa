# 🧠 Asistente Médico – API RAG + Agentes IA

API centralizada para la gestión inteligente de historias clínicas, turnos y consultas de patrimonio mediante **Agentes de IA y RAG (Retrieval-Augmented Generation)**.

---

## 🚀 Instalación

### Desarrollo Local

```bash
git clone https://gitlab.com/tu-usuario/asistente_medico_api.git
cd asistente_medico_api
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

### Producción (Recomendado - Docker)

#### Linux/Mac
```bash
./deploy.sh --docker
```

#### Windows
```bash
deploy.bat
```

### Producción (Docker Manual)

```bash
# Construcción optimizada para evitar errores de dependencias
DOCKER_BUILDKIT=1 docker build \
  --build-arg PIP_DEFAULT_TIMEOUT=1000 \
  --build-arg PIP_RETRIES=3 \
  -t asistente-medico-api .

docker run -d \
  --name asistente-medico-api \
  --env-file .env \
  -p 8000:8000 \
  --restart unless-stopped \
  asistente-medico-api
```

### Producción (Manual)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir --timeout 1000 --retries 3 -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🌐 Documentación interactiva (Swagger)
Una vez iniciado el servidor, accedé a:
👉 http://127.0.0.1:8000/docs
Desde ahí podés probar todos los endpoints con ejemplos.


---

# 📚 Documentación de Endpoints

## 🏥 1. Historia Clínica (RAG + Qdrant)

Endpoints para interactuar con la historia clínica del paciente.

### `POST /api/hc/ingesta-DataPatient`
Busca paciente en el sistema externo con parámetros flexibles y carga/actualiza su Historia Clínica en Qdrant.

**Body (JSON) - Ejemplo:**
```json
{
  "patientIdentification": "94715079",
  "patientTypeIdentificationId": 1
}
```

### `POST /api/hc/consultar-agent` (Recomendado)
**Nuevo Endpoint del Agente Clínico**. Utiliza un agente autónomo capaz de razonar, buscar en Qdrant y sintetizar respuestas complejas.

**Parámetros Query:**
- `stream` (bool): `true` para respuesta en streaming (SSE), `false` para JSON único.

**Body (JSON) - Ejemplo Completo:**
```json
{
  "user_id": "maria",
  "session_id": "turno-123",
  "paciente_id": "168249",
  "pregunta": "¿Cuál fue el motivo de ingreso?",
  "group": [ "CM;HISTORIA CLINICA", "INDICACIONES FARMACOLÓGICAS" ], // Opcional: Filtro por grupo
  "healthHistoryGroup": "CM;EVOLUCIÓN", // Opcional
  "procedureNumber": 211143,            // Opcional: Filtro por trámite
  "servicio_id": [ 26 ],                // Opcional: Lista de IDs de servicio
  "servicio": "MEDICINA (5º CATEDRA)",  // Opcional: Descripción del servicio
  "name": "MOTIVO CONSULTA INICIAL"     // Opcional: Filtro por nombre de sección
}
```

**Respuesta (Stream `text/event-stream`):**
Eventos SSE: `message` (tokens de texto), `reasoning` (pensamiento interno), `final` (JSON final).


---

## 📂 Estructura del proyecto

```bash
asistente_medico_api/
│
├── app/
│   ├── agents/          # configuración de agentes IA
│   ├── routes/          # Endpoints y routers
│   ├── services/        # Lógica principal y motores RAG
│   └── utils/           # Funciones auxiliares
│
├── main.py              # Entrada principal de FastAPI
├── .env                 # Variables de entorno
├── requirements.txt     # Dependencias
└── README.md            # Documentación
```

## 🐳 Variables de Entorno (.env)

```env
# Qdrant Configuration
QDRANT_URL=https://tu-qdrant-instance.com
QDRANT_API_KEY=tu-api-key
QDRANT_COLLECTION=hc_chat_db

# Ollama Configuration  
OLLAMA_BASE_URL=http://localhost:11434

# Other Configuration
HCI_DEBUG=0
SIMILARITY_TOP_K=12
SUMMARY_TOP_K=50
```

## ⚠️ Notas de uso
✅ Primero usar `/api/hc/ingesta-json-hci-completehealthhistory`
✅ Luego se puede consultar al agente con `/api/hc/consultar-agent`.




