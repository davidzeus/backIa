import os
import json
import re
import ollama
import redis
import datetime
from decimal import Decimal
from typing import Optional, List, Dict
from fastapi import HTTPException
import pytds


DIRECCION_SERVIDOR = os.getenv("PAT_SQL_HOST", "localhost")
NOMBRE_BASE_DATOS = os.getenv("PAT_SQL_DB", "test")
USUARIO_BD = os.getenv("PAT_SQL_USER", "sa")
CONTRASENA_BD = os.getenv("PAT_SQL_PASS", "")

# Configuración de Redis
REDIS_HOST = os.getenv("PAT_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("PAT_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("PAT_REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("PAT_REDIS_PASSWORD", None)
REDIS_TTL = int(os.getenv("PAT_REDIS_TTL", "86400"))  # 24 horas por defecto


# --------------------------
# 2. MAPA DE LA BASE DE DATOS (TABLAS)
# --------------------------
MAPA_TABLAS_DB = {
    "v_inventario": "v_inventario(Orden int, CodPres int, Caracteristicas varchar, Marca nvarchar, Modelo nvarchar, Nserie nvarchar, ValorOrigen money, FechaAlta smalldatetime, Responsable nvarchar, Ubicacion varchar, Baja nvarchar, FechaBaja smalldatetime, Observaciones nvarchar, Libro varchar, Folio varchar, Elemento nvarchar)",
}

CAMPOS_TABLA = {
    "elemento": "Elemento: descripcion del tipo de elemento ",
    "ubicación": "Ubicacion: nombre del servicio en el que esta ubicado el activo o patrimonio",
    "orden": "Orden: nro de patrimonio",
    "caracteristicas": "Caracteristicas: descripcion mas completa del elemento aveces contine el modelo y marca",
    "codPres": "CodPres: número de código presupuestario que puede estar asociado a varios registros",
    "marca": "Marca: marca del elemento",
    "modelo": "Modelo: modelo del elemento",
    "nserie": "Nserie: número de serie del elemento ",
    "valorOrigen": "ValorOrigen: valor de adquisición del activo",
    "fechaAlta": "FechaAlta: e.g. fechaAlta=30/8/2019",
    "responsable": "Responsable: nombre del responsable del activo o patrimonio",
    "fechaBaja": "FechaBaja: es null si esta activo",
    "baja": "Baja: comentario de la baja",
    "observaciones": "observaciones: obeservaciones del patrimonio",
    "libro": "libro: numero correspondiente a un libro de actas físico",
    "folio": "folio: numero de folio en el que se registro",
}

REGLAS_ADICIONALES = {
    "add1": "Cuando consultes por descripciones de elementos o ubicaciones haz el where like '%%' para traer todos los registros que contengan ese valor. e.g. query : 'busca todos los monitores de la division cardiologia' then where sql: 'where elemento like '%monitor%' and ubicacion like %cardiologia% '",
    "add2": "Siempre que generes una columna calculada o una función de agregación (COUNT, SUM, AVG, MAX, MIN), debes asignar un alias descriptivo utilizando la palabra clave AS y comillas simples. Por ejemplo: COUNT(*) AS 'Cantidad'"
}


# --------------------------
# 3. GESTOR DE HISTORIAL REDIS
# --------------------------
class GestorHistorialRedis:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Verificar conexión
            self.redis_client.ping()
            print(f"✓ Conectado a Redis en {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            print(f"⚠ No se pudo conectar a Redis: {e}")
            self.redis_client = None

    def _get_key(self, id_usuario: str, id_session: str) -> str:
        """Genera la clave de Redis para el historial"""
        return f"patrimony_chat_history:{id_usuario}:{id_session}"

    def guardar_mensaje(self, id_usuario: str, id_session: str, rol: str, contenido: str):
        """Guarda un mensaje en el historial"""
        if not self.redis_client:
            return False
        
        try:
            key = self._get_key(id_usuario, id_session)
            mensaje = {
                "role": rol,
                "content": contenido,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Agregar mensaje al historial
            self.redis_client.rpush(key, json.dumps(mensaje))
            
            # Establecer TTL
            self.redis_client.expire(key, REDIS_TTL)
            
            return True
        except Exception as e:
            print(f"Error al guardar mensaje en Redis: {e}")
            return False

    def obtener_historial(self, id_usuario: str, id_session: str, limite: int = 10) -> List[Dict]:
        """Obtiene el historial de conversación"""
        if not self.redis_client:
            return []
        
        try:
            key = self._get_key(id_usuario, id_session)
            mensajes = self.redis_client.lrange(key, -limite, -1)
            
            historial = []
            for msg in mensajes:
                mensaje = json.loads(msg)
                # Solo devolver role y content para Ollama
                historial.append({
                    "role": mensaje["role"],
                    "content": mensaje["content"]
                })
            
            return historial
        except Exception as e:
            print(f"Error al obtener historial de Redis: {e}")
            return []

    def limpiar_historial(self, id_usuario: str, id_session: str) -> bool:
        """Limpia el historial de una sesión"""
        if not self.redis_client:
            return False
        
        try:
            key = self._get_key(id_usuario, id_session)
            self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Error al limpiar historial: {e}")
            return False
    



# --- 4. CEREBRO LÓGICO ---

class GestorConsultasIA:
    def __init__(self, nombre_modelo, ollama_host):
        self.nombre_modelo = nombre_modelo
        self.ollama_host = ollama_host
        self.gestor_historial = GestorHistorialRedis()
        
        # print(f"Configurando cliente de Ollama con host: {ollama_host}")
        self.client = ollama.Client(host=ollama_host)

    def conectar_base_datos(self):
        try:
            
            # separar DIRECCION_SERVIDOR en servidor y instancia
            servidor, instancia = DIRECCION_SERVIDOR.split('\\')

            #print(f"Intentando conectar a {servidor} en la instancia {instancia}...")
            return pytds.connect(
                server=servidor + '\\' + instancia, 
                database=NOMBRE_BASE_DATOS, 
                user=USUARIO_BD, 
                password=CONTRASENA_BD,
                validate_host=False 
            )
            
        except Exception as error:
            raise RuntimeError(f"Error al conectar a la base de datos: {error}")

    def traducir_pregunta_a_sql_stream(
        self, 
        modo_stream: bool,
        texto_pregunta: str, 
        nombre_tabla: str,
        id_usuario: Optional[str] = None,
        id_session: Optional[str] = None,
        usar_historial: bool = False
    ):
        """Genera SQL en modo stream usando ollama.chat() y devuelve un generador"""
        texto_esquemas = "\n".join([f"- {valor}" for clave, valor in MAPA_TABLAS_DB.items()])
        texto_rel = "\n".join([f"- {valor}" for clave, valor in CAMPOS_TABLA.items()])
        texto_add = "\n".join([f"- {valor}" for clave, valor in REGLAS_ADICIONALES.items()])
        
        system_prompt = (
            "Eres un experto Ingeniero de Datos SQL Server (T-SQL).\n"
            "Genera consultas compatibles con T-SQL antiguo. REGLA CRÍTICA: No uses jamás 'OFFSET' ni 'FETCH NEXT'. Para limitar resultados, utiliza siempre 'SELECT TOP n'.\n"
            "Tu objetivo es traducir preguntas en lenguaje natural a consultas SQL ejecutables y precisas.\n\n"
            "La Base de datos pertenece a un sistema llamado Patrimonio en el que la empresa tiene registrados todos sus activos fisicos.\n\n"
            "--- BASE DE DATOS (ESTRUCTURA) ---\n"
            f"{texto_esquemas}\n\n"
            "--- DESCRIPCION DE CAMPOS ---\n"
            f"{texto_rel}\n\n"
            "--- REGLAS ADICIONALES ---\n"
            f"{texto_add}\n\n"
            "--- REGLAS OBLIGATORIAS ---\n"
            "1. Responde SOLO con código SQL. Nada de texto extra.\n"
            "2. Usa sintaxis SQL Server (T-SQL).\n"
            "3. Usa 'TOP n' en vez de 'LIMIT' o 'OFFSET'. La estructura debe ser: SELECT TOP {n} {columnas} FROM {tabla} GROUP BY {columnas} ORDER BY {columna} DESC.\n"
            "4. NO usar OFFSET n ROWS FETCH NEXT n ROWS ONLY\n"
            "5. NO usar 'LIMIT n' o 'OFFSET n'.\n"
            "6. Usa 'GETDATE()' para fechas.\n"
            "7. Prioriza el uso de `LIKE` para búsquedas de texto parcial.\n"
            "8. Para campos de fecha, usa el formato `YYYY-MM-DD`.\n"
            "9. Si no puedes generar una consulta, devuelve un comentario SQL con el motivo (ej: -- No puedo responder a eso).\n"
            "10. Si el usuario hace referencia a consultas anteriores (ej: 'lo mismo pero...', 'agregale...'), usa el contexto de mensajes previos para entender qué modificar."
        )

        user_message = (
            "--- PETICIÓN DEL USUARIO ---\n"
            f"Tabla principal: {nombre_tabla}\n"
            f"Pregunta: \"{texto_pregunta}\""
        )

        # Construir mensajes con historial si está habilitado
        messages = [{'role': 'system', 'content': system_prompt}]
        
        if usar_historial and id_usuario and id_session:
            historial = self.gestor_historial.obtener_historial(id_usuario, id_session)
            # print(f"Historial recuperado de Redis: {historial}")
            messages.extend(historial)
        
        messages.append({'role': 'user', 'content': user_message})

        # Guardar el mensaje del usuario en Redis
        if usar_historial and id_usuario and id_session:
            self.gestor_historial.guardar_mensaje(id_usuario, id_session, 'user', user_message)

        try:
            # print(f"Enviando consulta a Ollama (historial: {usar_historial})...")
            
            stream = self.client.chat(
                model=self.nombre_modelo,
                messages=messages,
                stream=modo_stream,
                options={'temperature': 0.1},
                think=False
            )
            
            return stream

        except Exception as error:
            print(f"Error al obtener respuesta de Ollama: {error}")
            raise RuntimeError(f"Error al obtener respuesta de Ollama: {error}")

    def limpiar_sql(self, codigo_sql):
        """Limpia el código SQL generado"""
        codigo_sql = re.sub(r'<think>.*?</think>', '', codigo_sql, flags=re.DOTALL)
        codigo_sql = codigo_sql.replace("```sql", "").replace("```", "").strip()
        
        lineas_validas = [
            linea for linea in codigo_sql.splitlines() 
            if not linea.strip().startswith(("--", "Nota:", "Note:"))
        ]
        
        return "\n".join(lineas_validas).strip()

    def verificar_seguridad_sql(self, codigo_sql):
        if not codigo_sql:
            return False, "La IA no generó ningún código."

        sql_en_mayusculas = codigo_sql.strip().upper()
        palabras_prohibidas = ["DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE", "ALTER", "GRANT", "REVOKE", "EXEC"]

        if not sql_en_mayusculas.startswith("SELECT") and not sql_en_mayusculas.startswith("WITH"):
            return False, "Por seguridad, solo se permiten lecturas (SELECT)."

        for palabra in palabras_prohibidas:
            if re.search(r'\b' + palabra + r'\b', sql_en_mayusculas):
                return False, f"Comando prohibido detectado: {palabra}"

        return True, None

    def ejecutar_consulta_en_bd(self, codigo_sql):
        conexion = None
        try:
            conexion = self.conectar_base_datos()
            cursor = conexion.cursor()
            cursor.execute(codigo_sql)

            if cursor.description:
                nombres_columnas = [col[0] for col in cursor.description]
                filas = cursor.fetchall()
                
                resultado = []
                for row in filas:
                    fila_dict = dict(zip(nombres_columnas, row))
                    for clave, valor in fila_dict.items():
                        if isinstance(valor, Decimal):
                            fila_dict[clave] = str(valor)
                        elif isinstance(valor, (datetime.date, datetime.datetime)) and valor is not None:
                            fila_dict[clave] = valor.isoformat()
                    resultado.append(fila_dict)

                return resultado
            else:
                return {"info": "Consulta ejecutada sin devolver datos"}
            
        except Exception as error:
            raise RuntimeError(f"Error al ejecutar la consulta en la base de datos: {error}")
        finally:
            if conexion:
                try:
                    conexion.close()
                except:
                    pass

    def generate_response_no_stream(
        self, 
        response, 
        id_usuario: Optional[str] = None,
        id_session: Optional[str] = None,
        usar_historial: bool = False
    ):
        sql_generado = response['message']['content']
        # print(f"SQL crudo:\n{sql_generado}")
        
        # Guardar respuesta del asistente en Redis
        if usar_historial and id_usuario and id_session:
            self.gestor_historial.guardar_mensaje(id_usuario, id_session, 'assistant', sql_generado)
        
        codigo_sql = self.limpiar_sql(sql_generado)
        # print(f"SQL generado:\n{codigo_sql}")
        
        if not codigo_sql:
            raise HTTPException(status_code=400, detail="La IA no pudo generar la consulta.")

        es_seguro, mensaje_error = self.verificar_seguridad_sql(codigo_sql)
        if not es_seguro:
            raise HTTPException(status_code=403, detail=f"Consulta no permitida: {mensaje_error}")

        resultado = self.ejecutar_consulta_en_bd(codigo_sql)
        return resultado

    async def generate_response_stream(
        self, 
        stream,
        id_usuario: Optional[str] = None,
        id_session: Optional[str] = None,
        usar_historial: bool = False
    ):
        codigo_sql_completo = []
        inicio = time.perf_counter()
            
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                token = chunk['message']['content']
                # print(f"token: {token}")

                if token:
                    codigo_sql_completo.append(token)
                    yield json.dumps({"type": "token", "content": token}) + "\n"
            
            if chunk.get('done', False):
                fin = time.perf_counter()
                # print(f"Streaming completado en {fin - inicio:.2f} segundos")
                break
        
        sql_completo = "".join(codigo_sql_completo)
        # print(f"respuesta IA:\n{sql_completo}")
        
        # Guardar respuesta del asistente en Redis
        if usar_historial and id_usuario and id_session:
            self.gestor_historial.guardar_mensaje(id_usuario, id_session, 'assistant', sql_completo)
        
        codigo_sql_limpio = self.limpiar_sql(sql_completo)
        # print(f"SQL generado:\n{codigo_sql_limpio}")
        
        es_seguro, mensaje_error = self.verificar_seguridad_sql(codigo_sql_limpio)
        
        if not es_seguro:
            yield json.dumps({
                "type": "error",
                "content": f"Consulta no permitida: {mensaje_error}"
            }) + "\n"
            return
        
        try:
            resultado = self.ejecutar_consulta_en_bd(codigo_sql_limpio)
            yield json.dumps({
                "type": "query_result",
                "content": resultado
            }) + "\n"
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": f"Error al ejecutar consulta: {str(e)}"
            }) + "\n"

    
    def interpretar_resultado_con_ia(self, pregunta_usuario: str, resultado_json: str, historial: List[Dict], stream: bool = False):
        """Usa la IA para interpretar un resultado JSON y responder en lenguaje natural."""
        
        system_prompt = (
            "Eres un asistente de datos amigable y servicial. Tu tarea es interpretar los resultados de una consulta (en formato JSON) y responder a la pregunta original del usuario de forma clara y concisa en lenguaje natural."
            "No menciones que recibiste un JSON, solo responde la pregunta."
        )

        user_message = (
            f"Pregunta del usuario: '{pregunta_usuario}'\n\n"
            f"Datos para responder (en formato JSON):\n{resultado_json}"
        )

        messages = [{'role': 'system', 'content': system_prompt}]
        if historial:
            messages.extend(historial)
        messages.append({'role': 'user', 'content': user_message})

        try:
            response = self.client.chat(
                model=self.nombre_modelo,
                messages=messages,
                stream=stream
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Error al interpretar el resultado con la IA: {e}")


