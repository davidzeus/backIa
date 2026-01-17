import logging
import os
import re
from typing import Dict, Optional
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup

from app.decorators.http_error_handler import manejar_errores_httpx


@manejar_errores_httpx("obtención de historia clínica por trámite")
async def obtener_historial_texto(id_tramite: int) -> str | None:
    """
    Devuelve el texto limpio de la HC o None si el trámite no existe.
    """
    camas_back = os.getenv("CAMAS_BACK", "http://10.10.18.35:3333/")
    url = f"{camas_back.rstrip('/')}/api/hospitalization/hc/summary/{id_tramite}"
    timeout = httpx.Timeout(30.0, connect=60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        logging.info(f"📡  GET {url}")
        try:
            resp = await client.get(url)
            if resp.status_code in (400, 404):
                logging.warning(
                    f"⚠️  Trámite {id_tramite} inexistente (HTTP {resp.status_code})"
                )
                return None  # ←  indicador de «no existe»
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # otros códigos 5xx, 401, etc. los delega al decorador
            raise e

        # ─── Limpieza de HTML ───
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "head", "title", "meta", "link"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()

        if not text:
            raise ValueError("El contenido de la historia clínica está vacío")

        logging.info(f"✅  HC obtenida para trámite {id_tramite}")
        return text


@manejar_errores_httpx("obtención de historia clínica por paciente")
async def obtener_json_por_paciente(patient_id: int) -> Dict:
    """
    Obtiene el historial clínico completo de un paciente en formato JSON.

    Args:
        patient_id (int): ID del paciente

    Returns:
        Dict: Historia clínica completa en formato JSON
    """
    hci_back = os.getenv("HCI_BACK", "http://10.10.18.35:8888/")
    url = f"{hci_back.rstrip('/')}/api/hcl/completehealthhistory"

    payload = {
        "HC": {},
        "patientId": patient_id,
        "serviceId": 0,
        "endDate": None,
        "specialtyId": 0,
    }

    timeout = httpx.Timeout(30.0, connect=60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        logging.info(f"Obteniendo historia clínica para paciente {patient_id}")
        response = await client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        logging.info(
            f"Historia clínica obtenida exitosamente para paciente {patient_id}"
        )
        return data


@manejar_errores_httpx("búsqueda de pacientes")
async def buscar_pacientes(
    dni: Optional[int] = None,
    apellido: Optional[str] = None,
    nombre: Optional[str] = None,
    hc: Optional[str] = None,
    sexo: Optional[str] = None,
    edad: Optional[int] = None,
    provincia: Optional[str] = None,
    estado_civil: Optional[str] = None,
    obra_social: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict:
    base_url = "http://10.10.18.35:4444/api/patient/"

    params = {}
    if dni:
        params["patientIdentification"] = dni
    if apellido:
        params["patientLastName"] = apellido
    if nombre:
        params["patientName"] = nombre
    if hc:
        params["patientHC"] = hc
    if sexo:
        params["patientSex"] = sexo
    if edad:
        params["patientAge"] = edad
    if provincia:
        params["province"] = provincia
    if estado_civil:
        params["maritalStatus"] = estado_civil
    if obra_social:
        params["patientHealthCoverageDescription"] = obra_social
    if email:
        params["patientEmail"] = email

    query = urlencode(params)
    url = f"{base_url}?{query}"

    timeout = httpx.Timeout(30.0, connect=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        logging.info(f"Buscando pacientes con parámetros: {params}")
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
