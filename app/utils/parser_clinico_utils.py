import json
from pathlib import Path
from typing import Dict, List


def parsear_json_clinico(json_data: dict) -> List[Dict]:
    """Parsea un JSON clínico y devuelve una lista de textos estructurados por sección"""
    secciones = []

    entrys = json_data.get("entrys", [])
    for entrada in entrys:
        service = entrada.get("service", {}).get("description", "Servicio desconocido")
        doctor = entrada.get("doctor", {})
        medico = f"{doctor.get('lastName', '')}, {doctor.get('firstName', '')}".strip(
            ", "
        )
        fecha_ingreso = entrada.get("dateHour") or entrada.get("date") or ""
        hc_id = entrada.get("id", "Sin ID")

        estructuras = entrada.get("structureWithRecords", [])
        for estructura in estructuras:
            group = estructura.get("group", "Sin grupo")
            name = estructura.get("name", "Sin nombre")
            updated_at = estructura.get("updatedAt", "")

            for record in estructura.get("records", []):
                texto = record.get("value", "")
                if not texto or len(texto.strip()) < 20:
                    continue

                fecha_registro = record.get("dateTime") or updated_at or fecha_ingreso

                contenido = f"""🩺 Servicio: {service}
👨‍⚕️ Médico: {medico}
🗂️ Grupo: {group}
📄 Sección: {name}
📅 Fecha: {fecha_registro}
📝 Texto: 
{texto.strip()}"""

                secciones.append(
                    {
                        "hc_id": hc_id,
                        "group": group,
                        "name": name,
                        "fecha": fecha_registro,
                        "medico": medico,
                        "servicio": service,
                        "texto": contenido,
                    }
                )

    return secciones


def probar_parser_desde_txt(path_archivo: str):
    ruta = Path(path_archivo)
    if not ruta.exists():
        print(f"❌ Archivo no encontrado: {ruta}")
        return

    with open(ruta, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    resultados = parsear_json_clinico(json_data)
    if not resultados:
        print("⚠️ No se generaron resultados.")
    else:
        print(f"✅ Se generaron {len(resultados)} secciones:")
        for i, r in enumerate(
            resultados[:3]
        ):  # Mostramos solo las primeras 3 para verificar
            print(f"\n--- Sección {i+1} ---")
            print(r["texto"])


# 🔧 Llamada de ejemplo (puede ir al final del archivo o en un notebook/script de prueba)
if __name__ == "__main__":
    probar_parser_desde_txt("hc_test.json")
