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

### `POST /api/hc/consultar-qdrant_hci` (Legacy)
Búsqueda RAG directa sin razonamiento complejo. Más rápido para búsquedas simples.

---

## 📅 2. Agente de Turnos (Appointments)

### `GET /api/agent/appointment/chat`
Permite interactuar en tiempo real con el agente de IA para consultas sobre turnos médicos. La respuesta se transmite en formato de eventos (SSE).

**Parámetros:**
- `message` (str): El mensaje o pregunta del usuario para el agente.
- `user_id` (str, opcional): Identificador único del usuario (memoria persistente).
- `session_id` (str, opcional): Identificador único de la sesión de chat (memoria persistente).

**Filtros de Búsqueda (Query String):**
Se pueden agregar filtros a la búsqueda de turnos médicos utilizando parámetros de consulta en la URL. 
Ejemplo: `/api/appointment/chat?message=...&patientId=123&serviceId=456`.

**Response (JSON):**
```json
{
  "data": "respuesta",
  "session_info": {
        "session_id": "1",
        "user_id": "1"
    }
}
```

### `GET /api/agent/appointment/chat/stream`
Versión streaming del chat de turnos. Devuelve un stream de eventos (SSE) donde cada evento contiene un fragmento de la respuesta.

**Response (SSE):**
```json
{
  "data": "respuesta"
}
```

---

## 🏛️ 3. Agente de Patrimonio

### `GET /api/agent/patrimony/chat`
Permite realizar consultas al agente de IA sobre patrimonio.

**Parámetros:**

- `message` (str): El mensaje o pregunta del usuario para el agente.
- `user_id` (str, opcional): Identificador único del usuario(memoria persistente).
- `session_id` (str, opcional): Identificador único de la sesión de chat(memoria persistente).

Filtros de Búsqueda (Query String):
Se pueden agregar filtros a la búsqueda sobre el inventario de patrimonio utilizando parámetros de consulta en la URL.
**Ejemplo:** `/api/agent/patrimony/chat?message=...&codPres=437`

Configuración del Agente:
Todas las instrucciones y el comportamiento del agente de IA pueden ser modificados a través de un archivo de configuración YAML ubicado en agents/agno/patrimonio, permitiendo una personalización flexible sin cambios en el código.


**Response:**
```json
{
  "data": "respuesta",
  "session_info": {
        "session_id": "1",
        "user_id": "1"
    }
}
```

### `GET /api/patrimony/chat-json`
Genera y ejecuta una consulta SQL a partir de una pregunta en lenguaje natural, devolviendo el resultado en formato JSON estructurado.

**Parámetros:**

- `message` (str): La pregunta en lenguaje natural.
- `tabla_principal` (str): Tabla o vista a consultar (default: v_inventario).
- `usar_historial` (bool): Habilita el uso del historial de conversación. Historial de Redis (default: False)
- `id_session` (str): ID de la sesión (requerido si usar_historial=True).
- `id_usuario` (str): ID del usuario (requerido si usar_historial=True).

**Ejemplo Request:**
`GET /api/patrimony/chat-json?message=ultimos 2 patrimonios registrado en endocrino&id_usuario=patrimonyUser&id_session=session123&usar_historial=true`

**Response:**
```json
{
    "data": [
        {
            "Número de Patrimonio": 23089,
            "Tipo de Activo": "IMPRESORA LASER ",
            "Fecha de Alta": "2023-01-31",
            "Marca": null,
            "Modelo": null,
            "Número de Serie": null,
            "Valor de Adquisición": "45000.0000",
            "Ubicación": "DIVISION ENDOCRINOLOGIA"
        },
        {
            "Número de Patrimonio": 23066,
            "Tipo de Activo": "CONVECTOR VICTRO",
            "Fecha de Alta": "2022-11-30",
            "Marca": null,
            "Modelo": null,
            "Número de Serie": null,
            "Valor de Adquisición": "27966.5200",
            "Ubicación": "DIVISION ENDOCRINOLOGIA"
        }
    ]

}
```

### `GET /api/patrimony/chat` (SQL + Interpretación)
Genera una consulta SQL, la ejecuta y la IA interpreta el resultado en lenguaje natural. Soporta streaming.

**Parámetros:**
- `message` (str).
- `stream` (bool): Habilita streaming SSE.

**Response (sin streaming):**
```json
{
    "data": "El último patrimonio registrado... corresponde a una impresora láser..."
}
```
Response (con streaming)

Content-Type: text/event-stream
```json
{"data": "El"}
{"data": " último"}
{"data": " patrimonio"}
```


Response error
```json
{
    "error": "Consulta no permitida: Por seguridad, solo se permiten lecturas (SELECT).",
    "status_code": 403
}
```

El historial de conversación se almacena en Redis y permite mantener el contexto entre múltiples consultas.
El parámetro stream=true utiliza Server-Sent Events (SSE) para respuestas en tiempo real.

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

## 🔧 Solución de Problemas

### Error "resolution-too-deep" en Docker
Si obtienes errores de dependencias al construir la imagen Docker:

```bash
# Opción 1: Usar pip con configuración de timeout
docker build --build-arg PIP_DEFAULT_TIMEOUT=1000 -t asistente-medico-api .

# Opción 2: Construir sin cache
docker build --no-cache -t asistente-medico-api .

# Opción 3: Usar buildkit
DOCKER_BUILDKIT=1 docker build -t asistente-medico-api .
```

### Problemas de memoria durante instalación
```bash
pip install --no-cache-dir -r requirements.txt --verbose --timeout 1000
```

## ⚠️ Notas de uso
✅ Primero usar `/api/hc/ingesta-json-hci-completehealthhistory` o `/api/hc/procesar`.
✅ Luego se puede consultar o resumir con los otros endpoints.
✅ El campo `json_data` se obtiene desde el sistema hospitalario.



