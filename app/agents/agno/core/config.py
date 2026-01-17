import yaml
from pathlib import Path
from typing import Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Clase para manejar la configuración de la aplicación"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración desde archivo YAML"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Validar configuración requerida
            self._validate_config(config)
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validar que la configuración tenga los campos requeridos"""
        required_fields = ['model', 'agent']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Campo requerido faltante en configuración: {field}")
    
    def get(self, key: str, default=None):
        """Obtener valor de configuración con soporte para claves anidadas"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value