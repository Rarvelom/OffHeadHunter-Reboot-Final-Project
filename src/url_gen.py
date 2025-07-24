from urllib.parse import quote
from typing import Union


def encode_job_title(job_title: str) -> str:
    return quote(job_title.strip())

def generar_url_infojobs(puesto: str,
                         modalidad: str = None,
                         salario_minimo: int = None,
                         localidades: Union[str, list] = None) -> str:
    """
    Genera una URL de búsqueda para InfoJobs con los parámetros especificados.
    
    Args:
        puesto: Título del puesto de trabajo
        modalidad: Modalidad de trabajo ("Presencial", "Híbrido", "A distancia")
        salario_minimo: Salario mínimo anual deseado (bruto)
        localidades: Lista de localidades o provincia (ej: "Barcelona" o ["Barcelona", "Madrid"])
        
    Returns:
        str: URL formateada para la búsqueda en InfoJobs
    """
    # Validar que el puesto no esté vacío
    if not puesto or not puesto.strip():
        raise ValueError("El puesto no puede estar vacío")
    
    puesto_enc = encode_job_title(puesto.strip())

    # Mapeo de modalidades a sus IDs en InfoJobs
    modalidad_ids = {
        "Presencial": "1",
        "Híbrido": "2",
        "A distancia": "3"
    }
    
    # Normalizar la modalidad
    modalidad_id = None
    if modalidad:
        modalidad = modalidad.strip()
        modalidad_id = modalidad_ids.get(modalidad.capitalize())

    # Normalizar localidades
    if isinstance(localidades, list):
        localidades = ",".join(str(loc).strip() for loc in localidades if str(loc).strip())
    elif isinstance(localidades, str):
        localidades = localidades.strip().strip('"')
    
    # Validar que al menos un parámetro de búsqueda esté presente
    if not any([puesto, modalidad_id, salario_minimo, localidades]):
        raise ValueError("Se requiere al menos un criterio de búsqueda")

    # Construcción de parámetros base
    params = {
        "keyword": puesto_enc,
        "page": "1",
        "sortBy": "RELEVANCE",
        "onlyForeignCountry": "false",
        "sinceDate": "ANY"
    }

    # Añadir parámetros opcionales si están definidos y no vacíos
    if modalidad_id:
        params["teleworkingIds"] = modalidad_id
    if salario_minimo and str(salario_minimo).isdigit():
        params["salaryMin"] = str(salario_minimo)
        params["salaryPeriod"] = "YEAR"
        params["salaryType"] = "GROSS"
    if localidades:
        # Si son códigos postales o IDs de provincia, asegurarse de que sean números
        try:
            # Si es un número, asumimos que es un ID de provincia
            if localidades.isdigit():
                params["provinceIds"] = localidades
            # Si es un nombre de ciudad, lo dejamos como está para que InfoJobs lo interprete
            else:
                params["provinceIds"] = localidades
        except (AttributeError, ValueError):
            # Si hay algún error, intentamos usar el valor tal cual
            params["provinceIds"] = localidades

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    return f"https://www.infojobs.net/ofertas-trabajo?{query_string}"