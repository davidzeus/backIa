from agno.tools.sql import SQLTools
from sqlalchemy import create_engine
import os
from agno.tools import tool
import httpx
from typing import Optional

HOST_API_PATRIMONIO = os.getenv("HOST_API_PATRIMONIO", "localhost")
   
@tool    
def consultar_patrimonio(
    elemento: Optional[str] = None,
    ubicacion: Optional[str] = None,
    orden: Optional[int] = None,
    caracteristicas: Optional[str] = None,
    codPres: Optional[int] = None,
    marca: Optional[str] = None,
    modelo: Optional[str] = None,
    nserie: Optional[str] = None,
    valorOrigen: Optional[float] = None,
    fechaAlta: Optional[str] = None,
    responsable: Optional[str] = None,
    fechaBaja: Optional[str] = None,
    observaciones: Optional[str] = None,
    libro: Optional[int] = None,
    folio: Optional[int] = None,
    limit: int = 10
) -> str:
    """
    Consulta el inventario del sistema Patrimonio.
    
    Args:
        elemento: Descripción del elemento a buscar
        ubicacion: Descripción del servicio/ubicación
        orden: Número de orden o patrimonio
        limit: Cantidad máxima de resultados (default: 10)
        ... (otros parámetros)
    
    Returns:
        str: JSON con los resultados de la consulta
    """
    
    # Construir parámetros de query
    params = {"limit": limit}
    if elemento: params["elemento"] = elemento
    if ubicacion: params["ubicacion"] = ubicacion
    if orden: params["orden"] = orden
    if caracteristicas: params["caracteristicas"] = caracteristicas
    if codPres: params["codPres"] = codPres
    if marca: params["marca"] = marca
    if modelo: params["modelo"] = modelo
    if nserie: params["nserie"] = nserie
    if valorOrigen: params["valorOrigen"] = valorOrigen
    if fechaAlta: params["fechaAlta"] = fechaAlta
    if responsable: params["responsable"] = responsable
    if fechaBaja: params["fechaBaja"] = fechaBaja
    if observaciones: params["observaciones"] = observaciones
    if libro: params["libro"] = libro
    if folio: params["folio"] = folio
    
    try:
        response = httpx.get(f"{HOST_API_PATRIMONIO}/api/patrimony", params=params, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error consultando API: {str(e)}"

def sql_tool():

    PAT_SQL_HOST = os.getenv("PAT_SQL_HOST")
    PAT_SQL_USER = os.getenv("PAT_SQL_USER")
    PAT_SQL_PASS = os.getenv("PAT_SQL_PASS")
    PAT_SQL_DB = os.getenv("PAT_SQL_DB")

    if not (PAT_SQL_HOST and PAT_SQL_USER and PAT_SQL_PASS and PAT_SQL_DB):
        raise ValueError("URL de base de datos no configurada")

    db_url = f"mssql+pyodbc://{PAT_SQL_USER}:{PAT_SQL_PASS}@{PAT_SQL_HOST}/{PAT_SQL_DB}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    print(db_url)
    if not db_url:
        raise ValueError("URL de base de datos no configurada")

    db_engine = create_engine(db_url)

    return SQLTools(db_engine=db_engine)

