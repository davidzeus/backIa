import logging
from functools import wraps

import httpx


def manejar_errores_httpx(nombre_operacion: str):
    """
    Decorador para manejar errores comunes de httpx de manera uniforme.

    Args:
        nombre_operacion (str): Nombre descriptivo de la operación para los logs

    Returns:
        Callable: Decorador configurado
    """

    def decorador(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except httpx.TimeoutException:
                msg = f"Timeout durante {nombre_operacion}"
                logging.error(msg)
                raise ValueError(msg)
            except httpx.HTTPStatusError as e:
                msg = f"Error HTTP {e.response.status_code} durante {nombre_operacion}"
                logging.error(msg)
                raise ValueError(msg)
            except httpx.RequestError as e:
                msg = f"Error de red durante {nombre_operacion}: {str(e)}"
                logging.error(msg)
                raise ValueError(msg)
            except Exception as e:
                msg = f"Error inesperado durante {nombre_operacion}: {str(e)}"
                logging.error(msg)
                raise ValueError(msg)

        return wrapper

    return decorador
