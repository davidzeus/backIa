# 🧠 Asistente Médico – RAG + LlamaIndex + FastAPI

Este proyecto es una **API en FastAPI** que permite procesar historias clínicas utilizando técnicas de **RAG (Retrieval-Augmented Generation)**.  
Combina un motor de consulta inteligente (**Query Engine**) con modelos LLM locales vía **Ollama** para generar resúmenes clínicos automáticos y responder preguntas clínicas específicas.

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
```bash
# Linux/Mac
./deploy.sh --docker

# Windows
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
🌐 Documentación interactiva (Swagger)
Una vez iniciado el servidor, accedé a:

👉 http://127.0.0.1:8000/docs

Desde ahí podés probar todos los endpoints con ejemplos.

# 📬 Endpoints disponibles

## `POST /api/hc/consultar-qdrant_hci`

Realiza una **pregunta clínica** sobre la historia indexada en Qdrant, aplicando **filtros dinámicos** (por paciente y/o número de trámite, más filtros opcionales).

---

### Ejemplo — “Motivo de ingreso” (paciente + trámite)

**Request**
```json
{
  "paciente_id": "163234",
  "pregunta": "¿Cuál fue el motivo de ingreso?",
  "procedureNumber": 181734
}
```
Response (resumen)


```json
{
  "paciente_id": "163234",
  "pregunta": "¿Cuál fue el motivo de ingreso?",
  "respuesta": "INFECCION EN SITIO QX",
  "sources": ["..."]
}
```
Valida

Todos los sources son de paciente_id = 163234 ✅

Todos los sources tienen procedureNumber = 181734 ✅

Evidencia textual: "MOTIVO CONSULTA INICIAL: INFECCION EN SITIO QX" ✅

Ejemplo — Consulta combinada (paciente + trámite + sección + grupo + servicio)
Request


```json
{
  "pregunta": "¿Cuál fue el motivo de ingreso?",
  "paciente_id": "163234",
  "procedureNumber": 181734,
  "name": "MOTIVO CONSULTA INICIAL",
  "group": "CM;HISTORIA CLINICA",
  "servicio_id": [59]
}
```
Response (resumen)


```json
{
  "paciente_id": "163234",
  "pregunta": "¿Cuál fue el motivo de ingreso?",
  "respuesta": "INFECCION EN SITIO QX",
  "sources": [
    {
      "text": "… MOTIVO CONSULTA INICIAL: INFECCION EN SITIO QX",
      "score": 0.13516554,
      "metadata": {
        "paciente_id": "163234",
        "procedureNumber": 181734,
        "name": "MOTIVO CONSULTA INICIAL",
        "seccion": "MOTIVO CONSULTA INICIAL",
        "seccion_raiz": "MOTIVO CONSULTA INICIAL",
        "group": "CM;HISTORIA CLINICA",
        "servicio_id": 59,
        "servicio_desc": "CIRUGIA GASTROENTEROLOGICA",
        "fecha": "2024-01-18",
        "profesional": null
      }
    }
  ]
}
```
Qué filtra este ejemplo

paciente_id = "163234" AND

procedureNumber = 181734 AND

name = "MOTIVO CONSULTA INICIAL" AND

group = "CM;HISTORIA CLINICA" AND

servicio_id ∈ [59] (OR interno si hubiera más IDs en la lista)

Valida

Fuentes del paciente 163234 ✅

Fuentes del trámite 181734 ✅

Coincidencia con sección y grupo solicitados ✅

Respuesta sustentada por el texto: “MOTIVO CONSULTA INICIAL: INFECCION EN SITIO QX” ✅

Tip: combinar paciente_id + procedureNumber + name maximiza la precisión cuando buscás motivos/diagnósticos puntuales.

Ejemplo — Extraer diagnósticos (paciente + sección)
Request

```json
{
  "pregunta": "Extrae los diagnósticos",
  "paciente_id": "163234",
  "name": "DIAGNÓSTICO"
}
```
Response (resumen)

```json
{
  "paciente_id": "163234",
  "pregunta": "Extrae los diagnósticos",
  "respuesta": "Cefalea, Síndrome del colon irritable con diarrea",
  "sources": [
    {
      "text": "… DIAGNÓSTICO: cefalea",
      "metadata": {
        "paciente_id": "163234",
        "name": "DIAGNÓSTICO",
        "group": "RECETA Y PRESCRIPCION",
        "fecha": "2025-08-14"
      }
    },
    {
      "text": "… DIAGNÓSTICO: Síndrome del colon irritable con diarrea",
      "metadata": {
        "paciente_id": "163234",
        "name": "DIAGNÓSTICO",
        "group": "HISTORIA CLINICA",
        "fecha": "2025-01-19"
      }
    }
  ]
}
```
Qué filtra este ejemplo

paciente_id = "163234" AND

name = "DIAGNÓSTICO" (recupera nodos cuya sección es DIAGNÓSTICO; puede provenir de distintos grupos/servicios)

Valida

Todas las fuentes pertenecen al paciente 163234 ✅

Todas las fuentes están en la sección DIAGNÓSTICO ✅

La respuesta concatena diagnósticos textuales encontrados: “Cefalea”, “Síndrome del colon irritable con diarrea” ✅

Tip: para diagnósticos por internación, agregá procedureNumber; para restringir por área, agregá group o servicio_id.

----------------------------------------------------------------------------
## `POST /api/hc/consultar-qdrant_hci/stream`

Request: Content-Type: application/json

Response: Content-Type: text/event-stream; charset=utf-8

Parámetros (body JSON)
```json
{
  "pregunta": "Texto de la consulta",
  "user_id": "opcional-para-memoria",
  "session_id": "opcional-para-memoria",
  "paciente_id": "string | requerido",
  "procedureNumber": 123456,          // opcional
  "group": "opcional",
  "healthHistoryGroup": "opcional",
  "servicio_id": 82 | [82, 15],       // opcional (número o lista)
  "servicio": "opcional",
  "name": "opcional"
}
```

El motor detecta intención temporal (“últimos”, “primeros”, rangos) y aplica ventana de tiempo cuando corresponde.

### Qué envía el stream

Durante la conexión, recibirás eventos SSE línea a línea. El formato es JSON por evento:

Tokens de texto (stream de la respuesta):

{"type":"token", "text":"…fragmento…"}


Cierre con resultado final (texto completo + fuentes top-k):
```
{
  "type":"end",
  "final": {
    "answer": "…respuesta completa…",
    "sources": [
      {
        "text": "…fragmento fuente…",
        "score": 0.57,
        "metadata": {
          "paciente_id": "163234",
          "procedureNumber": "",
          "name": "DIAGNÓSTICO",
          "seccion": "DIAGNÓSTICO",
          "seccion_raiz": "DIAGNÓSTICO",
          "group": "HISTORIA CLINICA",
          "healthHistoryGroup": null,
          "servicio_id": 82,
          "servicio_desc": "OFTALMOLOGIA",
          "fecha": "2025-08-29",
          "fecha_ts": 1756436400,
          "profesional": "APELLIDO, NOMBRE"
        }
      }
    ]
  }
}
```

Errores (si ocurre algo no recuperable):

{"type":"error","message":"…detalle…"}


Si el contexto filtrado no tiene coincidencias, el answer final puede ser:
⚠️ No hay contexto del paciente solicitado en la base (o el filtro no encontró coincidencias).

Ejemplo (cURL)
```
curl -N -X POST \
  -H "Content-Type: application/json" \
  http://<HOST>:<PORT>/api/hc/consultar-qdrant_hci/stream \
  -d '{
    "pregunta": "Extrae los diagnósticos del paciente",
    "user_id": "Nancy",
    "session_id": "turno-123",
    "paciente_id": "163234"
  }'
```

Salida típica (recortada):
```
data: {"type":"token","text":"Síndrom"}
data: {"type":"token","text":"e del colon irritable …"}
data: {"type":"end","final":{"answer":"Síndrome del colon irritable…","sources":[…]}}
```
### Integración frontend (Angular + sse.js)
### // service
```
postSseStream(url: string, body: any): Observable<string> {
  return new Observable(observer => {
    const source = new SSE(url, {
      headers: { 'Content-Type': 'application/json' },
      payload: JSON.stringify(body),
      method: 'POST'
    });
    source.addEventListener('message', (ev: any) => observer.next(ev.data));
    source.addEventListener('error', (err: any) => {
      if (source.readyState === 0) observer.complete();
      else observer.error(err);
      source.close();
    });
    source.stream();
    return () => source.close();
  });
}
```
### // componente (consumo)
```
this.sseService.postSseStream(endpoint, body).subscribe({
  next: (raw) => {
    const evt = JSON.parse(raw);
    if (evt.type === 'token') {
      this.buffer += evt.text;              // mostrar parcial
    } else if (evt.type === 'end') {
      this.buffer = evt.final.answer;       // reemplazar por completo
      this.sources = evt.final.sources;     // mostrar fuentes
    } else if (evt.type === 'error') {
      this.error = evt.message;
    }
  },
  complete: () => { /* stream cerrado */ },
  error: (e) => { /* manejar error */ }
});
```
Notas de uso

**Routing QA/Resumen**: si la pregunta contiene palabras como “resumen”, “síntesis”, “sumario”, se usa un sintetizador de resumen; si no, QA extractivo.

**Guard-rails**: el retrieval está filtrado por paciente_id/procedureNumber. Si llega algún nodo de otro paciente y STRICT_FILTER está activo, el stream aborta con error.

**Temporalidad**: el sistema detecta superlativos temporales (“últimos 2”, “primeros 3”) y rangos, y ajusta el retrieval.

**CORS/Proxies**: al ser SSE, el servidor responde text/event-stream. Si hay Nginx/Proxy, asegurá proxy_buffering off; para no romper el flujo.

**Time-outs**: clientes deben permitir conexiones largas.

Diferencias con el endpoint no stream

**No stream** (POST /api/hc/consultar-qdrant_hci): devuelve un JSON único:

{ "paciente_id": "...", "pregunta": "...", "respuesta": "…", "sources": [ … ] }


**Stream**: envía múltiples eventos token y un end con el mismo answer/sources finales.

## `GET /api/appointment/chat` 
💬 Chat con el Agente IA
Permite interactuar en tiempo real con el agente de IA para consultas sobre turnos médicos. La respuesta se transmiteen formato de eventos (SSE).

# 💬 Chat con el Agente IA
## GET /api/agent/appointment/chat
Permite realizar consultas al agente de IA sobre los turnos.

Descripción:
Este endpoint utiliza un agente de IA para responder preguntas relacionadas con turnos médicos. La comunicación se realiza a través de Server-Sent Events (SSE), lo que permite una transmisión de respuesta en tiempo real.

Parámetros:
  - `message` (str): El mensaje o pregunta del usuario para el agente.
  - `user_id` (str, opcional): Identificador único del usuario(memoria persistente).
  - `session_id` (str, opcional): Identificador único de la sesión de chat(memoria persistente).

Filtros de Búsqueda (Query String):
Se pueden agregar filtros a la búsqueda de turnos médicos utilizando parámetros de consulta en la URL. Estos filtros corresponden a los campos disponibles en la vista `[API].[V_Appointment]`. Por ejemplo: `/api/appointment/chat?message=...&patientId=123&serviceId=456`.

Configuración del Agente:
Todas las instrucciones y el comportamiento del agente de IA pueden ser modificados a través de un archivo de configuración YAML ubicado en agents/agno/appointments, permitiendo una personalización flexible sin cambios en el código.


Response 
```json

{
  "data": "respuesta",
  "session_info": {
        "session_id": "1",
        "user_id": "1"
    }
}

```

## GET /api/agent/appointment/chat/stream
Permite interactuar en tiempo real con el agente de IA para consultas sobre turnos médicos. La respuesta se transmite en formato de eventos (SSE).

Descripción:
Este endpoint utiliza un agente de IA para responder preguntas relacionadas con turnos médicos. La comunicación se realiza a través de Server-Sent Events (SSE), lo que permite una transmisión de respuesta en tiempo real.

Devuelve:
Un stream de eventos (SSE) donde cada evento contiene un fragmento de la respuesta del agente de IA. El contenido de la respuesta se encuentra dentro de `data:`.
Un stream de eventos (SSE) donde cada evento contiene un fragmento de la respuesta del agente de IA. 

Response 
```json

{
  "data": "respuesta"
}

```

## GET /api/agent/patrimony/chat
Permite realizar consultas al agente de IA sobre patrimonio.

Descripción:
Este endpoint utiliza un agente de IA para responder preguntas relacionadas con el patrimonio.

Parámetros:
  - `message` (str): El mensaje o pregunta del usuario para el agente.
  - `user_id` (str, opcional): Identificador único del usuario(memoria persistente).
  - `session_id` (str, opcional): Identificador único de la sesión de chat(memoria persistente).

Filtros de Búsqueda (Query String):
Se pueden agregar filtros a la búsqueda sobre el inventario de patrimonio utilizando parámetros de consulta en la URL. Por ejemplo: `/api/agent/patrimony/chat?message=...&codPres=437`.

Configuración del Agente:
Todas las instrucciones y el comportamiento del agente de IA pueden ser modificados a través de un archivo de configuración YAML ubicado en agents/agno/patrimonio, permitiendo una personalización flexible sin cambios en el código.

Response 
```json

{
  "data": "respuesta",
  "session_info": {
        "session_id": "1",
        "user_id": "1"
    }
}

```
# 💬 Consulta Base Patrimonio
## GET /api/parimony/chat-json
Genera y ejecuta una consulta SQL a partir de una pregunta en lenguaje natural, devolviendo el resultado en formato JSON.

Descripción:
Este endpoint recibe una pregunta en lenguaje natural y genera una consulta SQL utilizando. Luego, ejecuta la consulta y devuelve el resultado en formato JSON.

Parámetros:
  - `message` (str): La pregunta en lenguaje natural.
  - `tabla_principal` (str): Tabla o vista a consultar (default: v_inventario).
  - `usar_historial` (bool): Habilita el uso del historial de conversación. Historial de Redis (default: False)
  - `id_session` (str): ID de la sesión (requerido si usar_historial=True).
  - `id_usuario` (str): ID del usuario (requerido si usar_historial=True).
  
Ejemplo de Solicitud
```bash
GET /api/patrimony/chat-json?message=ultimos 2 patrimonios registrado en endocrino&id_usuario=patrimonyUser&id_session=session123&usar_historial=true
```
Response 
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

## GET /api/parimony/chat
Genera una consulta, la ejecuta y la IA interpreta el resultado.

Recibe una pregunta, la traduce a SQL, ejecuta la consulta y devuelve una interpretación en lenguaje natural. 
Si stream=True, la respuesta se devuelve en tiempo real.

Parámetros:
  - `message` (str): La pregunta en lenguaje natural.
  - `tabla_principal` (str): Tabla o vista a consultar (default: v_inventario).
  - `usar_historial` (bool): Habilita el uso del historial de conversación. Historial de Redis (default: False)
  - `id_session` (str): ID de la sesión (requerido si usar_historial=True).
  - `id_usuario` (str): ID del usuario (requerido si usar_historial=True).
  - `stream` (bool): Habilita el streaming de la respuesta (formato de eventos SSE) (default: False).

#### Ejemplo de Solicitud
```bash
GET /api/patrimony/chat?message=ultimo patrimonio registrado en endocrino&stream=true
```
#### Response (sin streaming)
```json
Content-Type: application/json
{
    "data": "El último patrimonio registrado en el área de Endocrino corresponde a una impresora láser adquirida el **31 de enero de 2023**, con un valor original de **45.000 unidades monetarias** (valor de origen). Actualmente, su valor registrado es de **23.089 unidades**."
}
```
#### Response (con streaming)
```json
Content-Type: text/event-stream

{"data": "El"}
{"data": " último"}
{"data": " patrimonio"}
```
#### Response error
```json
{
    "error": "Consulta no permitida: Por seguridad, solo se permiten lecturas (SELECT).",
    "status_code": 403
}
```
El historial de conversación se almacena en Redis y permite mantener el contexto entre múltiples consultas.
El parámetro stream=true utiliza Server-Sent Events (SSE) para respuestas en tiempo real.


📂 Estructura del proyecto
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

## 🐳 Variables de Entorno

Crear un archivo `.env` con las siguientes variables:

```bash
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

# Opción 3: Usar buildkit para mejor resolución de dependencias
DOCKER_BUILDKIT=1 docker build -t asistente-medico-api .
```

### Problemas de memoria durante instalación
```bash
# Aumentar memoria disponible para pip
pip install --no-cache-dir -r requirements.txt --verbose --timeout 1000
```

---

## ⚠️ Notas de uso
✅ Primero usar /api/hc/ingesta-json-hci-completehealthhistory o /api/hc/procesar.
✅ Luego se puede consultar o resumir con los otros endpoints.
✅ El campo json_data se obtiene desde el sistema hospitalario:
📥 http://10.10.18.35:8888/api/hci/completehealthhistory