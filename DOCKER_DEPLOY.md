# 🏥 Asistente Médico API - Despliegue con Docker

Esta guía te ayu## 🏗️ Arquitectura de Servicios

El despliegue incluye los siguientes servicios:

| Servicio | Puerto | Ubicación | Descripción |
|----------|--------|-----------|-------------|
| **asistente-medico-api** | 8000 | Docker | API principal FastAPI |
| **qdrant** | 6333, 6334 | Externo (10.10.0.48) | Base de datos vectorial |
| **ollama** | 11434 | Externo (10.10.0.48) | Servidor de modelos LLM |
| **sqlserver** | 1433 | Docker | Base de datos SQL Server |

### 🔧 Configuraciones disponibles:

- **Producción**: Solo API + SQL Server en Docker, Qdrant y Ollama externos en 10.10.0.48
- **Desarrollo**: Misma configuración, solo la aplicación en Dockerplegar el Asistente Médico API en tu servidor de producción usando Docker y Docker Compose.

## 📋 Requisitos Previos

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **4GB RAM mínimo** (recomendado 8GB) - Reducido ya que solo corre API + SQL Server
- **5GB espacio libre** en disco (sin modelos LLM ni Qdrant, ya que están en servidor externo)
- **Servidor Qdrant externo** funcionando en `http://10.10.0.48:6333`
- **Servidor Ollama externo** funcionando en `http://10.10.0.48:11434`
- **Modelo médico** `thewindmom/llama3-med42-8b` descargado en el servidor Ollama
- **Conectividad de red** hacia el servidor 10.10.0.48

### 🔧 Configuración de servicios externos

Asegúrate de que:
1. **Qdrant** esté corriendo en `10.10.0.48:6333`
2. **Ollama** esté corriendo en `10.10.0.48:11434`
3. El modelo esté descargado: `ollama pull thewindmom/llama3-med42-8b`
4. El firewall permita conexiones a los puertos 6333 y 11434

### 🧪 Verificación previa

Antes del despliegue, verifica la conectividad:

```bash
# Verificar Qdrant
curl http://10.10.0.48:6333/health

# Verificar Ollama
curl http://10.10.0.48:11434/api/tags

# Verificar modelo médico
curl http://10.10.0.48:11434/api/tags | grep "thewindmom/llama3-med42-8b"
```

## 🚀 Despliegue Rápido

1. **Clona el repositorio** (si no lo has hecho):
```bash
git clone <tu-repositorio>
cd asistente_medico_api
```

2. **Ejecuta el script de despliegue**:
```bash
./deploy.sh
```

El script automáticamente:
- Verifica las dependencias
- Construye las imágenes Docker
- Levanta todos los servicios
- Descarga los modelos necesarios
- Verifica que todo esté funcionando

### 🔄 Configuraciones de Despliegue

#### Producción (Todos los servicios en Docker)
```bash
# Usar docker-compose.yml para producción completa
docker-compose up -d --build
```

#### Desarrollo (Configuración híbrida)
```bash
# Usar docker-compose.dev.yml para desarrollo
docker-compose -f docker-compose.dev.yml up -d --build
```

El archivo `docker-compose.dev.yml` incluye:
- ✅ **Qdrant**: Local en Docker
- ⚙️ **Ollama**: Externo por defecto (configurable)
- ⚙️ **SQL Server**: Externo por defecto (configurable)

## 🛠️ Despliegue Manual

Si prefieres hacer el despliegue paso a paso:

### 1. Preparar el entorno
```bash
# Crear directorios necesarios
mkdir -p model_cache datasets

# Configurar permisos
chmod 755 model_cache datasets
```

### 2. Configurar variables de entorno
El archivo `.env` ya está configurado para Docker. Las variables principales son:
- `LLM_BASE_URL=http://ollama:11434`
- `QDRANT_URL=http://qdrant:6333`
- `HOST=0.0.0.0`
- `DB_HOST=sqlserver`

### 3. Levantar los servicios
```bash
# Construir y levantar
docker-compose up -d --build

# Ver logs
docker-compose logs -f
```

### 4. Descargar modelos de IA
```bash
# Descargar el modelo médico (esto puede tardar 10-15 minutos)
docker-compose exec ollama ollama pull thewindmom/llama3-med42-8b
```

## 🏗️ Arquitectura de Servicios

El despliegue incluye los siguientes servicios:

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| **asistente-medico-api** | 8000 | API principal FastAPI |
| **qdrant** | 6333, 6334 | Base de datos vectorial |
| **ollama** | 11434 | Servidor de modelos LLM |
| **sqlserver** | 1433 | Base de datos SQL Server |

## 🔗 URLs de Acceso

Una vez desplegado, puedes acceder a:

- **API Principal**: http://tu-servidor:8000
- **Documentación API**: http://tu-servidor:8000/docs
- **Dashboard Qdrant**: http://10.10.0.48:6333/dashboard (servidor externo)
- **API Ollama**: http://10.10.0.48:11434 (servidor externo)

## 📊 Monitoreo y Logs

```bash
# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f asistente-medico-api
docker-compose logs -f ollama
docker-compose logs -f qdrant
docker-compose logs -f sqlserver

# Ver estado de los servicios
docker-compose ps

# Ver uso de recursos
docker stats
```

## 🔧 Mantenimiento

### Reiniciar servicios
```bash
# Reiniciar todo
docker-compose restart

# Reiniciar servicio específico
docker-compose restart asistente-medico-api
```

### Actualizar la aplicación
```bash
# Parar servicios
docker-compose down

# Obtener últimos cambios
git pull

# Reconstruir y levantar
docker-compose up -d --build
```

### Hacer backup de datos
```bash
# Backup de Qdrant
docker-compose exec qdrant tar -czf /tmp/qdrant-backup.tar.gz /qdrant/storage
docker cp $(docker-compose ps -q qdrant):/tmp/qdrant-backup.tar.gz ./qdrant-backup.tar.gz

# Backup de SQL Server
docker-compose exec sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P 'AsistenteMedico2024!' -Q "BACKUP DATABASE AsistenteMedico TO DISK = '/tmp/asistente_medico.bak'"
docker cp $(docker-compose ps -q sqlserver):/tmp/asistente_medico.bak ./sqlserver-backup.bak
```

### Limpiar datos (⚠️ CUIDADO)
```bash
# Parar y eliminar contenedores y volúmenes
docker-compose down -v

# Eliminar imágenes no utilizadas
docker image prune -f
```

## ⚡ Optimización para Producción

### Para servidores con GPU
Descomenta las líneas en `docker-compose.yml`:
```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

### Para servidores con mucha RAM
Ajusta las variables en `.env`:
```env
ITEM_TOKENS_MAX=4000
ITEM_MAX_CONCURRENT=5
```

### Para múltiples workers
Modifica el comando en `Dockerfile`:
```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## 🚨 Solución de Problemas

### El servicio no arranca
```bash
# Verificar logs
docker-compose logs asistente-medico-api

# Verificar conectividad de red
docker-compose exec asistente-medico-api ping qdrant
docker-compose exec asistente-medico-api ping ollama
```

### Problema con modelos de IA
```bash
# Limpiar y redescargar modelos
docker-compose exec ollama ollama rm thewindmom/llama3-med42-8b
docker-compose exec ollama ollama pull thewindmom/llama3-med42-8b
```

### Problemas de memoria
```bash
# Ver uso de memoria
docker stats

# Ajustar límites en docker-compose.yml
services:
  asistente-medico-api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Puerto ocupado
```bash
# Verificar puertos en uso
sudo netstat -tulpn | grep :8000

# Cambiar puerto en docker-compose.yml
ports:
  - "8080:8000"  # Cambiar puerto externo
```

## 🔒 Seguridad

Para producción, considera:

1. **Cambiar contraseñas por defecto**:
   - SQL Server: `SA_PASSWORD=AsistenteMedico2024!`
   
2. **Usar HTTPS** con un proxy reverso (nginx/traefik)

3. **Firewall**: Solo exponer puertos necesarios

4. **Secrets**: Usar Docker Secrets para contraseñas

## 📞 Soporte

Si encuentras problemas:
1. Revisa los logs con `docker-compose logs -f`
2. Verifica que todos los servicios estén corriendo con `docker-compose ps`
3. Asegúrate de tener suficiente RAM y espacio en disco
4. Verifica que los puertos no estén ocupados
