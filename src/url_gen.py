import urllib.parse

# Mapeo entre modalidad en texto y su ID correspondiente
MODALIDAD_IDS = {
    "Presencial": 1,
    "A distancia": 2,
    "Híbrido": 3,
    "Sin especificar": 4
}

def generar_url_infojobs(puesto, modalidad=None):
    """
    Genera una URL de búsqueda en InfoJobs para el puesto y modalidad especificados.
    
    Parámetros:
        puesto (str): Texto a buscar (por ejemplo, 'Analista de datos').
        modalidad (str o list[str], opcional): Modalidad de trabajo. Puede ser un string ('presencial') o lista ['remoto', 'hibrido'].
    
    Retorna:
        str: URL completa de búsqueda.
    """
    keyword = urllib.parse.quote(puesto)
    
    # Construcción de parámetros base
    params = {
        "keyword": keyword,
        "segmentId": "",
        "page": 1,
        "sortBy": "PUBLICATION_DATE",
        "onlyForeignCountry": "false",
        "sinceDate": "ANY"
    }
    
    # Añadir modalidad si se especifica
    if modalidad:
        if isinstance(modalidad, str):
            modalidad = [modalidad]
        ids = [str(MODALIDAD_IDS[m.title()]) for m in modalidad if m.title() in MODALIDAD_IDS]
        if ids:
            params["teleworkingIds"] = ",".join(ids)
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"https://www.infojobs.net/ofertas-trabajo?{query_string}"