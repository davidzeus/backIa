@echo off
REM Script de despliegue para Windows
REM Asistente Médico API

echo 🏥 Iniciando despliegue del Asistente Médico API...

REM Verificar si Docker está instalado
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker no está instalado. Por favor instala Docker Desktop primero.
    pause
    exit /b 1
)

REM Crear directorios necesarios
echo 📁 Creando directorios necesarios...
if not exist "model_cache" mkdir model_cache
if not exist "datasets" mkdir datasets

REM Verificar si existe .env
if not exist ".env" (
    echo ⚠️  Archivo .env no encontrado. Creando uno por defecto...
    (
    echo # Qdrant Configuration
    echo QDRANT_URL=http://10.10.0.48:6333
    echo QDRANT_API_KEY=
    echo QDRANT_COLLECTION=hc_chat_db
    echo.
    echo # Ollama Configuration  
    echo OLLAMA_BASE_URL=http://10.10.0.48:11434
    echo.
    echo # Backend Services
    echo HCI_BACK=http://10.10.18.35:8888/
    echo CAMAS_BACK=http://10.10.18.35:3333/
    echo.
    echo # Debug and Performance
    echo HCI_DEBUG=0
    echo SIMILARITY_TOP_K=12
    echo SUMMARY_TOP_K=50
    echo QDRANT_TIMEOUT=10.0
    ) > .env
    echo    ✅ Archivo .env creado. Por favor revisa y ajusta las configuraciones.
    pause
)

REM Construir imagen Docker con optimizaciones
echo 🐳 Construyendo imagen Docker...
set DOCKER_BUILDKIT=1
docker build --build-arg PIP_DEFAULT_TIMEOUT=1000 --build-arg PIP_RETRIES=3 -t asistente-medico-api .

if %errorlevel% neq 0 (
    echo ❌ Error al construir la imagen Docker.
    echo Intentando sin BuildKit...
    docker build --build-arg PIP_DEFAULT_TIMEOUT=1000 --build-arg PIP_RETRIES=3 -t asistente-medico-api .
    if %errorlevel% neq 0 (
        echo ❌ Error al construir la imagen. Verifica el archivo requirements.txt
        pause
        exit /b 1
    )
)

REM Detener contenedor anterior si existe
docker stop asistente-medico-api >nul 2>&1
docker rm asistente-medico-api >nul 2>&1

REM Ejecutar nuevo contenedor
echo 🚀 Iniciando contenedor...
docker run -d --name asistente-medico-api --env-file .env -p 8000:8000 --restart unless-stopped asistente-medico-api

if %errorlevel% equ 0 (
    echo 🎉 Despliegue completado exitosamente!
    echo 📡 API disponible en: http://localhost:8000
    echo 📚 Documentación en: http://localhost:8000/docs
    echo.
    echo Presiona cualquier tecla para abrir la documentación...
    pause >nul
    start http://localhost:8000/docs
) else (
    echo ❌ Error al iniciar el contenedor.
    pause
)