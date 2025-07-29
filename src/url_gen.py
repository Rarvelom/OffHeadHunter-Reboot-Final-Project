from urllib.parse import quote
from typing import Union


def encode_job_title(job_title: str) -> str:
    return quote(job_title.strip())

def generar_url_infojobs(puesto: str,
                         modalidad: str = None,
                         salario_minimo: int = None,
                         localidades: Union[str, list] = None) -> str:

    puesto_enc = encode_job_title(puesto)

    # Normaliza localidades
    if isinstance(localidades, list):
        localidades = ",".join(localidades)
    elif isinstance(localidades, str):
        localidades = localidades.strip().strip('"')

    # Construcción de parámetros base
    params = {
        "keyword": puesto_enc,
        "segmentId": "",
        "page": "1",
        "sortBy": "RELEVANCE",
        "onlyForeignCountry": "false",
        "sinceDate": "ANY"
    }

    # Añadir parámetros opcionales si están definidos y no vacíos
    if modalidad:
        params["teleworkingIds"] = modalidad
    if salario_minimo:
        params["salaryMin"] = salario_minimo
        params["salaryPeriod"] = "YEAR"
        params["salaryType"] = "GROSS"
    if localidades:
        params["provinceIds"] = localidades

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    return f"https://www.infojobs.net/ofertas-trabajo?{query_string}"