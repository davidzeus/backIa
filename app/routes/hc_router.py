# hc_router.py
import logging
import os
import unicodedata
import json
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, status, Query
from fastapi.concurrency import run_in_threadpool
from llama_index.core.base.llms.base import BaseLLM
from pydantic import BaseModel

from app.cache.engine_cache import invalidate_engine_cache_for_patient
from app.infra.ml_providers import get_llm
from app.schemas.esquema import ConsultaQdrantRequest
from app.services.ingesta_json_hci_completehealthhistory import (
    agrupar_items_usando_linea_de_tiempo,
    guardar_chunks,
)

from app.services.resumen_hci_completehealthhistory_por_seccion_service import (
    resumir_hc_por_seccion,
)

from app.services.query_engine_manager_service import get_router, query_hc, stream_hc
from sse_starlette.sse import EventSourceResponse
import time, json

from app.services.query_engine_manager_service import (
    get_router,          # dependency existente
    query_hc_stream,     # <- NUEVO: stream de eventos (token/end/error)
)
from app.services.agent_service import run_agent_consult, run_agent_consult_stream, delete_agent_session # 🆕 Servicio Agente


router = APIRouter(prefix="/api/hc", tags=["Historias Clínicas"])

# Logger del módulo (silencia INFO/DEBUG por defecto)
log = logging.getLogger(__name__)
if not bool(int(os.getenv("HCI_DEBUG", "0"))):  # exportá HCI_DEBUG=1 para ver debug
    log.setLevel(logging.WARNING)


#############################################################################
class PacienteInfo(BaseModel):
    id: int
    nombre: Optional[str] = None
    apellido: Optional[str] = None
    dni: Optional[str] = None


class IngestaResponse(BaseModel):
    status: str
    mensaje: str
    cantidad_chunks: int
    paciente_id: str
    document_id: Optional[str] = None
    inserted: Optional[int] = 0
    deleted: Optional[int] = 0
    change_ratio: Optional[float] = None
    no_op: Optional[bool] = None
    paciente: Optional[PacienteInfo] = None


@router.post(
    "/ingesta-DataPatient",
    summary="(POST) Busca paciente en (http://10.10.18.35:4444/api/patient/?patientIdentification=28073163&patientTypeIdentificationId=1) con parámetros flexibles y carga/actualiza su HC en Qdrant",
    response_model=IngestaResponse,
)
async def buscar_paciente_flexible(
    filtros: Dict[str, Any] = Body(
        ...,
        description=(
            "Parámetros para `GET /api/patient/` (4444). "
            "Enviá cualquier combinación válida, p.ej.: "
            '{"patientIdentification":"28073163","patientTypeIdentificationId":1}'
        ),
        example={"patientIdentification": "28073163", "patientTypeIdentificationId": 1},
    )
):
    try:
        # 1) Normalizar y validar filtros
        if not isinstance(filtros, dict) or not filtros:
            raise HTTPException(
                status_code=400, detail="Debe enviar al menos un parámetro de búsqueda."
            )
        search_params = {
            k: (v.strip() if isinstance(v, str) else v)
            for k, v in filtros.items()
            if v is not None and v != ""
        }
        if not search_params:
            raise HTTPException(
                status_code=400,
                detail="Todos los parámetros de búsqueda están vacíos o nulos.",
            )
        log.debug("🔎 Búsqueda 4444 (sanitizada): %s", search_params)

        # 2) Buscar paciente en 4444
        url_pac = "http://10.10.18.35:4444/api/patient/"
        async with httpx.AsyncClient(timeout=10) as client:
            resp_4444 = await client.get(url_pac, params=search_params)
            log.debug(
                "🌐 GET %s?%s -> %s",
                url_pac,
                resp_4444.request.url.query,
                resp_4444.status_code,
            )
            resp_4444.raise_for_status()
            data_4444 = resp_4444.json()

        if not data_4444.get("ok") or not data_4444.get("patients"):
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró paciente con esos datos. Parámetros: {search_params}",
            )

        paciente = data_4444["patients"][0]
        paciente_id_int = int(paciente["id"])
        paciente_id = str(paciente_id_int)

        # 3) Pedir HC al HCI (8888) – defaults seguros
        hci_back = os.getenv("HCI_BACK", "http://10.10.18.35:8888/")
        url_hc = f"{hci_back.rstrip('/')}/api/hci/completehealthhistory"
        body = {
            "patientId": paciente_id_int,
            "serviceId": 0,
            "specialtyId": 0,
            "startDate": None,
            "endDate": None,
            "HC": {},
            "procedureNumber": 0,
            "structureId": None,
            "medicId": None,
            "lastRecordOnly": False,
            "endpointId": None,
        }

        log.debug(
            "📤 POST %s | body=%s", url_hc, {k: v for k, v in body.items() if k != "HC"}
        )
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                url_hc,
                json=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            raw_text = r.text
            try:
                json_data = r.json()
            except ValueError:
                json_data = None

            if r.status_code >= 400:
                msg = (
                    str(json_data.get("msg", "")) if isinstance(json_data, dict) else ""
                )
                combined = f"{msg} {raw_text}".upper()
                if ("BH003" in combined) or ("EHH03" in combined):
                    log.warning(
                        "⚠️ HCI BH003/EHH03 | URL=%s | resp=%s",
                        r.request.url,
                        raw_text[:500],
                    )
                    raise HTTPException(
                        status_code=404,
                        detail="Paciente sin historia clínica disponible (BH003/EHH03).",
                    )
                log.error(
                    "⛔ 8888 %s | URL=%s | resp=%s",
                    r.status_code,
                    r.request.url,
                    raw_text[:500],
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Error del servicio de HC (8888): {r.status_code}. Resp: {raw_text[:300]}",
                )

            if isinstance(json_data, dict) and not json_data.get("ok", True):
                up = str(json_data.get("msg", "")).upper()
                if ("BH003" in up) or ("EHH03" in up):
                    raise HTTPException(
                        status_code=404,
                        detail="Paciente sin historia clínica disponible (BH003/EHH03).",
                    )

        # 4) Extraer contenido y agrupar
        contenido = _extraer_contenido_real(json_data)
        if not contenido.get("entrys"):
            raise HTTPException(
                status_code=404,
                detail="Paciente sin historia clínica registrada (entrys vacío).",
            )

        chunks = agrupar_items_usando_linea_de_tiempo(contenido, paciente_id)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron eventos clínicos válidos en la historia.",
            )

        # 5) Guardar en Qdrant (idempotente por paciente)
        res = guardar_chunks(chunks, paciente_id=paciente_id)
        if "error" in res:
            raise HTTPException(status_code=500, detail=res["error"])

        inserted = int(res.get("inserted", 0) or 0)
        deleted = int(res.get("deleted", 0) or 0)
        no_op = inserted == 0 and deleted == 0

        # 6) Invalidar caché por paciente solo si hubo cambios
        if not no_op:
            invalidate_engine_cache_for_patient(paciente_id)

        mensaje = (
            "La historia clínica ya estaba actualizada (sin cambios)."
            if no_op
            else "Historia clínica cargada/actualizada exitosamente."
        )

        # 7) Respuesta
        return IngestaResponse(
            status="ok",
            mensaje=mensaje,
            cantidad_chunks=len(chunks),
            paciente_id=paciente_id,
            document_id=res.get("document_id"),
            inserted=inserted,
            deleted=deleted,
            change_ratio=res.get("change_ratio"),
            no_op=no_op,
            paciente=PacienteInfo(
                id=paciente_id_int,
                nombre=paciente.get("firstName") or paciente.get("patientName"),
                apellido=paciente.get("lastName") or paciente.get("patientLastName"),
                dni=(
                    (paciente.get("identifier") or {}).get("number")
                    if isinstance(paciente.get("identifier"), dict)
                    else paciente.get("patientIdentification")
                ),
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("❌ Error en buscar_paciente_flexible")
        raise HTTPException(status_code=500, detail=str(e))


###############################
def _wrap_sse(gen):
    """
    Adapta el generador de eventos del servicio a SSE:
      - event: message  -> chunks parciales (data: {"data": "<chunk>"})
      - event: final    -> respuesta final (data: {"answer": "...", "sources": [...]})
      - event: error    -> errores controlados (data: {"error": "..."})
    """
    try:
        for ev in gen:
            t = ev.get("type")
            if t == "token":
                yield {
                    "event": "message",
                    "data": json.dumps({"data": ev.get("text", "")}, ensure_ascii=False),
                }
            elif t == "end":
                yield {
                    "event": "final",
                    "data": json.dumps(ev.get("final", {"answer": "", "sources": []}), ensure_ascii=False),
                }
            elif t == "error":
                yield {
                    "event": "error",
                    "data": json.dumps({"error": ev.get("message", "unknown")}, ensure_ascii=False),
                }
            # tipos desconocidos se ignoran
    except Exception as e:
        # fallback ante excepción en el generador
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}, ensure_ascii=False),
        }


async def _wrap_sse_async(gen):
    """
    Versión Async de _wrap_sse para generadores asíncronos (como el del Agente).
    """
    try:
        async for ev in gen:
            t = ev.get("type")
            if t == "token":
                yield {
                    "event": "message",
                    "data": json.dumps({"data": ev.get("text", "")}, ensure_ascii=False),
                }
            elif t == "reasoning_token":
                yield {
                    "event": "reasoning",
                    "data": json.dumps({"token": ev.get("text", "")}, ensure_ascii=False),
                }
            elif t == "meta":
                 yield {
                    "event": "meta",
                    "data": json.dumps({"status": ev.get("status", "")}, ensure_ascii=False),
                }
            elif t == "end":
                yield {
                    "event": "final",
                    "data": json.dumps(ev.get("final", {"answer": "", "sources": []}), ensure_ascii=False),
                }
            elif t == "error":
                yield {
                    "event": "error",
                    "data": json.dumps({"error": ev.get("message", "unknown")}, ensure_ascii=False),
                }
    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}, ensure_ascii=False),
        }


@router.post(
    "/consultar-qdrant_hci",
    summary="Pregunta sobre la historia clínica en Qdrant (JSON o SSE con stream=true)",
)
async def consultar_hc_qdrant(
    payload: ConsultaQdrantRequest,
    engine=Depends(get_router),
    stream: bool = Query(False, description="Si es true, responde por SSE"),
):
    """
    Un único endpoint:
      - JSON (por defecto) -> misma lógica y filtros que stream.
      - SSE si ?stream=true -> tokens por 'message' y cierre en 'final'.
    """
    try:
        # Normalización de entrada (igual para ambos modos)
        pregunta = unicodedata.normalize("NFKC", (payload.pregunta or "")).strip()
        if len(pregunta) < 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="La pregunta es demasiado corta.",
            )

        if engine is None:
            # Comportamiento coherente en ambos modos
            if stream:
                def _no_data():
                    yield {
                        "event": "message",
                        "data": json.dumps({"data": "⚠️ El paciente no tiene datos cargados."}, ensure_ascii=False),
                    }
                    yield {
                        "event": "final",
                        "data": json.dumps({"answer": "", "sources": []}, ensure_ascii=False),
                    }
                return EventSourceResponse(
                    _no_data(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontró historia clínica para el paciente especificado.",
            )

        # ───────────────────────────
        # MODO STREAM (SSE)
        # ───────────────────────────
        if stream:
            gen = stream_hc(pregunta, engine)
            return EventSourceResponse(
                _wrap_sse(gen),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        # ───────────────────────────
        # MODO JSON (sin stream)
        # ───────────────────────────
        # query_hc usa el mismo engine/filtros del stream
        resp: dict = await run_in_threadpool(query_hc, pregunta, engine)

        if not resp or not resp.get("answer"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sin contenido para responder con los datos disponibles.",
            )

        sentinel = resp["answer"].strip().lstrip(" \n\r\t")
        if sentinel.startswith("⚠️"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=resp["answer"],
            )

        return {
            "paciente_id": payload.paciente_id,
            "pregunta": pregunta,
            "respuesta": resp["answer"].strip(),
            "sources": resp["sources"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("❌ Error interno en consultar_hc_qdrant")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 🆕 ENDPOINT EXPERIMENTAL: AGENTE CLÍNICO (AGNO)
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/consultar-agent",
    summary="[BETA] Consulta usando el Agente Clínico Autónomo (Model: ministral-3:3b-instruct-2512-q8_0)",
)
async def consultar_agente(
    payload: ConsultaQdrantRequest, 
    stream: bool = Query(False, description="Activa modo streaming SSE")
):
    """
    Endpoint experimental que delega la lógica al Agente (agent_service).
    Soporta ?stream=true para respuesta token a token (SSE).
    """
    try:
        log.info(f"🤖 [agent] Consulta recibida (stream={stream})")
        
        if stream:
            # 🌊 Modo Stream SSE (Gen Async)
            gen = run_agent_consult_stream(payload)
            return EventSourceResponse(
                _wrap_sse_async(gen),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        # 🐢 Modo Síncrono (JSON)
        resultado = await run_agent_consult(payload)
        
        return {
            "paciente_id": payload.paciente_id,
            "pregunta": payload.pregunta,
            "respuesta": resultado.get("respuesta", ""),
            "sources": resultado.get("sources", []),
            "agent_model": os.getenv("LLM_MODEL_AGENT", "unknown")
        }
    except Exception as e:
        log.error(f"❌ [agent] Error en endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/agent-session/{session_id}",
    summary="Elimina una sesión del Agente de memoria/Redis",
)
async def eliminar_sesion_agente(
    session_id: str,
    patient_id: Optional[str] = Query(None, description="Si la sesión es específica de un paciente, enviar su ID para reconstruir la clave compuesta.")
):
    """
    Elimina permanentemente la sesión del Agente (memoria conversacional) de Redis.
    Útil para limpiar historial al cerrar una pestaña o finalizar consulta.
    """
    # Reconstrucción de clave compuesta si aplica
    target_session = session_id
    if patient_id:
        target_session = f"{session_id}::{patient_id}"
        log.info(f"🔑 Reconstruyendo clave compuesta: {target_session}")

    exito = delete_agent_session(target_session)
    
    if exito == 1:
        return {"status": "ok", "message": f"Sesión {target_session} eliminada.", "session_id": target_session}
    elif exito == 0:
        raise HTTPException(status_code=404, detail=f"Sesión {target_session} no encontrada.")
    else:
        raise HTTPException(status_code=500, detail="Error interno intentando eliminar la sesión.")


###############################################################################
def _buscar_entrys(data: dict) -> dict:
    """
    Busca recursivamente la clave 'entrys' en cualquier subnivel.
    Retorna el dict que contiene 'entrys' válido.
    """
    if not isinstance(data, dict):
        raise ValueError("El JSON debe ser un diccionario.")

    if "entrys" in data and isinstance(data["entrys"], list):
        return data

    for key, value in data.items():
        if isinstance(value, dict):
            try:
                return _buscar_entrys(value)
            except ValueError:
                continue

    raise ValueError("No se encontró la clave 'entrys' válida en el JSON enviado.")


#############
def _extraer_contenido_real(json_data: dict) -> dict:
    """
    Extrae y valida el contenido clínico del JSON.

    Args:
        json_data: Diccionario JSON de entrada

    Returns:
        dict: Contenido clínico validado

    Raises:
        ValueError: Si la estructura no es válida
    """
    if not isinstance(json_data, dict):
        raise ValueError("El JSON debe ser un diccionario")

    # Caso 1: 'entrys' en raíz
    if "entrys" in json_data:
        return json_data

    # Caso 2: Bajo 'clinicalData'
    if "clinicalData" in json_data:
        clinical_data = json_data["clinicalData"]
        if isinstance(clinical_data, dict) and "entrys" in clinical_data:
            return clinical_data
        raise ValueError("clinicalData presente pero con estructura inválida")

    # Caso 3: Búsqueda en primer nivel
    for key, value in json_data.items():
        if isinstance(value, dict) and "entrys" in value:
            return value

    raise ValueError("No se encontró la estructura esperada con 'entrys'")

'''
class EntradaHCJson(BaseModel):
    json_data: dict
    user_id: int


@router.post(
    "/resumen-por-seccion-json-hci-completehealthhistory",
    summary="Genera un resumen narrativo detallado y organizado por secciones, directo desde el JSON clínico",
)
async def generar_resumen_por_seccion(
    payload: EntradaHCJson,
    llm: BaseLLM = Depends(get_llm),  # ⬅️ Inyección de dependencia
):
    """
    Qué hace: Resume historias clínicas completas en JSON, generando narrativas por sección, ordenadas y coherentes.
    [ ... descripción igual ... ]
    """
    try:
        data = _extraer_contenido_real(payload.json_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extrayendo contenido: {e}")

    try:
        resumen = await resumir_hc_por_seccion(data, llm=llm)  # ⬅️ Pasamos el LLM
    except Exception as e:
        log.exception("❌ Error al generar resumen: %s", e)
        raise HTTPException(status_code=500, detail=f"Error al generar resumen: {e}")

    return {"user": payload.user_id, "resumen": resumen}
'''
