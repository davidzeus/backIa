from agno.tools import tool
import httpx
import os

HOST_API_APPOINTMENT = os.getenv("HOST_API_APPOINTMENT", "localhost")

@tool
def get_patient_appointments(string_query: str) -> str:
    """
    Obtiene los turnos médicos de un paciente desde la API del hospital de clínicas.
    
    Args:
        string_query (str): string query para la Api ejemplo string_query="patientId=1&serviceId=3&id=4"
        
    Returns:
        str: Información formateada de los turnos del paciente
    """
    try:
        # Realizar la consulta a la API
        url = f"{HOST_API_APPOINTMENT}/api/appointmentsRequest/?{string_query}"
        response = httpx.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("ok"):
            return f"Error en la API: {data.get('msg', 'Error desconocido')}"
        
        appointments = data.get("appointmentsRequest", [])
        
        if not appointments:
            return f"No se encontraron turnos"
        
        # Formatear la información para el modelo
        formatted_info = []
        formatted_info.append(f"TURNOS MÉDICOS ")
        formatted_info.append("=" * 50)
        
        for i, apt in enumerate(appointments, 1):
            patient = apt.get("patient", {})
            appointment = apt.get("appointment", {})
            service = apt.get("service", {})
            
            formatted_info.append(f"\n📅 TURNO #{i}")
            formatted_info.append(f"   Paciente: {patient.get('firstName', 'N/A')} {patient.get('lastName', 'N/A')}")
            formatted_info.append(f"   Fecha: {appointment.get('date', 'N/A')}")
            formatted_info.append(f"   Hora: {appointment.get('hour', 'N/A')}")
            formatted_info.append(f"   Servicio: {service.get('description', 'N/A')}")
            formatted_info.append(f"   Observaciones: {apt.get('observations', 'Sin observaciones')}")
            formatted_info.append(f"   Estado: {appointment.get('status', {}).get('id', 'Sin estado')}")
            
            # Información del médico si está disponible
            schedule = appointment.get("schedule", {})
            doctor = schedule.get("doctor", {})
            if doctor:
                formatted_info.append(f"   Médico: {doctor.get('registration', {})}")
            
        return "\n".join(formatted_info)
        
    except httpx.TimeoutException:
        return "Error: Tiempo de espera agotado al consultar la API"
    except httpx.RequestError as e:
        return f"Error de conexión: {str(e)}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"

@tool
def get_all_appointments(string_query: str) -> str:
    """
    Obtiene todos los turnos médicos disponibles en el hospital.
    
    Args:
        string_query (str): string query para la Api ejemplo string_query="patientId=1&serviceId=3&limit=5&order=date DESC" 
                            los filtros limit y order son filtros adicionales para controlar la cantidad y el orden de los resultados
                            limit: El número máximo de resultados de citas a devolver
                            order: El orden en que se ordenan los datos ejemplo: &order=date DESC ordena los datos por el campo date de forma desendente (ASC si se quiere ascendente)
        
    Returns:
        str: Información compacta de todos los turnos del hospital
    """
    try:
        url = f"{HOST_API_APPOINTMENT}/api/appointments/?{string_query}"
        response = httpx.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("ok"):
            return f"Error API: {data.get('msg', 'Error desconocido')}"
        
        appointments = data.get("appointments", [])
        
        if not appointments:
            return "No hay turnos en el sistema"
        
        # Formatear de manera compacta
        result = [f"TURNOS HOSPITAL ({len(appointments)} total):"]
        
        # Estadísticas rápidas por estado
        status_count = {}
        for apt in appointments:
            status = apt.get("status", {}).get("id", "?")
            status_count[status] = status_count.get(status, 0) + 1
        
        status_names = {"B":"Bloqueado", "D":"Disponible", "F":"Feriado", "R":"Reasignado", "U":"Ocupado", "V":"Vacaciones", "W":"Video"}
        stats = [f"{s}({status_names.get(s,s)}):{c}" for s, c in status_count.items()]
        result.append(f"Estados: {', '.join(stats)}")
        
        # Detalles compactos de cada turno
        for i, apt in enumerate(appointments[:20], 1):  # Limitar a 20 turnos
            schedule = apt.get("schedule", {})
            doctor = schedule.get("doctor", {})
            patient = apt.get("patient", {})
            
            # Línea compacta por turno
            date_time = f"{apt.get('date', 'N/A')} {apt.get('hour', 'N/A')}"
            doctor_name = f"Dr.{doctor.get('lastName', 'N/A')}"
            service = f"Servicio: {schedule.get('service', {}).get('description', 'N/A')[:30]}"  # Truncar
            specialty = f"Especialidad: {schedule.get('speciality', {}).get('description', 'N/A')[:30]}"  # Truncar
            status = apt.get('status', {}).get('id', '?')
            
            patient_info = ""
            if patient and patient.get("firstName"):
                patient_info = f" | Pac:{patient.get('lastName', '')},{patient.get('firstName', '')}"
            
            result.append(f"{i}. {date_time} | {doctor_name} | {service} | {specialty} | {status}{patient_info}")
        
        if len(appointments) > 20:
            result.append(f"... y {len(appointments) - 20} turnos más")
        
        return "\n".join(result)
        
    except httpx.TimeoutException:
        return "Error: Timeout API"
    except httpx.RequestError as e:
        return f"Error conexión: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"