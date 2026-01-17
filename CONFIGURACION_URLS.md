# Configuración de URLs para el Asistente Médico API

## 🔧 Variables de configuración importantes

### Servicios en Servidor Externo (10.10.0.48)
La aplicación está configurada para usar Qdrant y Ollama en un servidor externo.

**Configuración actual en .env:**
```
LLM_BASE_URL=http://10.10.0.48:11434
QDRANT_URL=http://10.10.0.48:6333
```

**Para docker-compose:**
```
LLM_BASE_URL=http://10.10.0.48:11434
QDRANT_URL=http://10.10.0.48:6333
```

### 📝 Personalizar URLs

Si tus servicios están en servidores diferentes, actualiza estas variables:

#### Servicios en otra IP
```bash
# Si los servicios están en otro servidor
LLM_BASE_URL=http://192.168.1.100:11434
QDRANT_URL=http://192.168.1.100:6333
```

#### Puertos personalizados
```bash
# Si usan puertos diferentes
LLM_BASE_URL=http://10.10.0.48:8080
QDRANT_URL=http://10.10.0.48:6334
```

#### Servicios en servidores separados
```bash
# Si están en servidores diferentes
LLM_BASE_URL=http://servidor-ollama:11434
QDRANT_URL=http://servidor-qdrant:6333
```

### 🔄 Aplicar cambios

1. **Para desarrollo local**: Edita el archivo `.env`
2. **Para Docker**: Edita las variables de entorno en `docker-compose.yml`
3. **Reinicia los servicios**: `docker-compose restart`

### 🧪 Verificar conectividad

```bash
# Verificar Qdrant desde tu servidor local
curl http://10.10.0.48:6333/health

# Verificar Ollama desde tu servidor local
curl http://10.10.0.48:11434/api/tags

# Verificar desde dentro del contenedor Docker
docker-compose exec asistente-medico-api curl http://10.10.0.48:6333/health
docker-compose exec asistente-medico-api curl http://10.10.0.48:11434/api/tags

# Verificar que el modelo esté disponible
curl http://10.10.0.48:11434/api/tags | grep "thewindmom/llama3-med42-8b"
```

### 🔧 Comandos útiles para el servidor 10.10.0.48

```bash
# Verificar estado de Qdrant
systemctl status qdrant
# o si corre en Docker:
docker ps | grep qdrant

# Verificar estado de Ollama
systemctl status ollama
# o si corre en Docker:
docker ps | grep ollama

# En el servidor, descargar el modelo
ollama pull thewindmom/llama3-med42-8b

# Listar modelos disponibles
ollama list
```
