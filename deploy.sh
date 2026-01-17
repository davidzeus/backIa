#!/bin/bash

# Script de inicialización para el Asistente Médico API
# Este script configura y despliega la aplicación en producción

set -e

echo "🏥 Iniciando despliegue del Asistente Médico API..."

# Cargar variables de entorno
if [ -f .env ]; then
    echo "📋 Cargando variables de entorno desde .env..."
    source .env
else
    echo "⚠️  Archivo .env no encontrado. Usando valores por defecto..."
    LLM_BASE_URL=${LLM_BASE_URL:-"http://10.10.0.48"}
    QDRANT_URL=${QDRANT_URL:-"http://10.10.0.48:6333"}
    HCI_BACK=${HCI_BACK:-"http://10.10.18.35:8888/"}
    CAMAS_BACK=${CAMAS_BACK:-"http://10.10.18.35:3333/"}
fi

# Verificar que Docker y Docker Compose están instalados
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Por favor instala Docker Compose primero."
    exit 1
fi

# Crear directorios necesarios
echo "📁 Creando directorios necesarios..."
mkdir -p model_cache
mkdir -p datasets

# Configurar permisos
echo "🔐 Configurando permisos..."
chmod 755 model_cache
chmod 755 datasets

# Descargar modelos de Ollama
echo "🤖 Configurando modelos de Ollama..."
echo "⚠️  Nota: Ollama corre externamente en ${LLM_BASE_URL}"
echo "   Verifica que el servidor Ollama esté funcionando antes de continuar"

# Verificar conectividad con servicios externos
echo "🔗 Verificando conectividad con servicios externos..."
echo "   - Ollama: ${LLM_BASE_URL}"
echo "   - Qdrant: ${QDRANT_URL}"
echo "   - HCI Backend: ${HCI_BACK}"
echo "   - Camas Backend: ${CAMAS_BACK}"

# Construir y levantar servicios
echo "🏗️  Construyendo y levantando servicios..."
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up -d

echo "⏳ Esperando que los servicios estén listos..."

# Esperar a que Qdrant (externo) esté listo
echo "🔍 Verificando Qdrant externo..."
QDRANT_HEALTH_URL="${QDRANT_URL}/health"
while ! curl -s "${QDRANT_HEALTH_URL}" > /dev/null; do
    echo "   Esperando Qdrant en ${QDRANT_URL}..."
    sleep 5
done
echo "✅ Qdrant externo está listo"

# Esperar a que Ollama (externo) esté listo
echo "🦙 Verificando Ollama externo..."
OLLAMA_TAGS_URL="${LLM_BASE_URL}/api/tags"
while ! curl -s "${OLLAMA_TAGS_URL}" > /dev/null; do
    echo "   Esperando Ollama en ${LLM_BASE_URL}..."
    sleep 5
done
echo "✅ Ollama externo está listo"

# Verificar backends adicionales
echo "🏥 Verificando HCI Backend..."
if curl -s "${HCI_BACK}" > /dev/null; then
    echo "✅ HCI Backend accesible en ${HCI_BACK}"
else
    echo "⚠️  HCI Backend no accesible en ${HCI_BACK}"
fi

echo "🛏️ Verificando Camas Backend..."
if curl -s "${CAMAS_BACK}" > /dev/null; then
    echo "✅ Camas Backend accesible en ${CAMAS_BACK}"
else
    echo "⚠️  Camas Backend no accesible en ${CAMAS_BACK}"
fi

# Verificar modelo de Ollama (externo)
echo "📥 Verificando modelo de Ollama externo..."
if curl -s "${OLLAMA_TAGS_URL}" | grep -q "thewindmom/llama3-med42-8b"; then
    echo "✅ Modelo ya está disponible en Ollama externo"
else
    echo "⚠️  El modelo 'thewindmom/llama3-med42-8b' no está disponible en Ollama externo"
    echo "   Asegúrate de descargarlo en el servidor Ollama: ollama pull thewindmom/llama3-med42-8b"
fi

# Verificar que la aplicación esté funcionando
echo "🔍 Verificando la aplicación..."
sleep 30
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ La aplicación está funcionando correctamente"
else
    echo "⚠️  La aplicación puede estar iniciando aún. Verifica los logs con: docker compose logs -f asistente-medico-api"
fi

echo "🎉 Despliegue completado!"
echo ""
echo "📋 Información de los servicios:"
echo "   - API FastAPI: http://localhost:8000"
echo "   - Documentación API: http://localhost:8000/docs"
echo "   - Qdrant Dashboard: ${QDRANT_URL}/dashboard"
echo "   - Ollama API: ${LLM_BASE_URL}/api"
echo "   - HCI Backend: ${HCI_BACK}"
echo "   - Camas Backend: ${CAMAS_BACK}"
echo ""
echo "📊 Para ver los logs:"
echo "   docker compose logs -f"
echo ""
echo "🛑 Para detener los servicios:"
echo "   docker compose down"
echo ""
echo "🔄 Para reiniciar:"
echo "   docker compose restart"
