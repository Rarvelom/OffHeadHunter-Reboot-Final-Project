import urllib.parse

# Mapeo entre modalidad en texto y su ID correspondiente
MODALIDAD_IDS = {
    "Presencial": 1,
    "A distancia": 2,
    "Híbrido": 3,
    "Sin especificar": 4
}

# Mapeo de provincias españolas y sus códigos
PROVINCIAS = {
    "A Coruña": 28,
    "Álava/Araba": 2,
    "Albacete": 3,
    "Alicante/Alacant": 4,
    "Almería": 5,
    "Asturias": 6,
    "Ávila": 7,
    "Badajoz": 8,
    "Barcelona": 9,
    "Burgos": 10,
    "Cáceres": 11,
    "Cádiz": 12,
    "Cantabria": 13,
    "Castellón/Castelló": 14,
    "Ceuta": 15,
    "Ciudad Real": 16,
    "Córdoba": 17,
    "Cuenca": 18,
    "Girona": 19,
    "Granada": 21,
    "Guadalajara": 22,
    "Guipúzcoa/Gipuzkoa": 23,
    "Huelva": 24,
    "Huesca": 25,
    "Islas Baleares/Illes Balears": 26,
    "Jaén": 27,
    "La Rioja": 29,
    "Las Palmas": 20,
    "León": 30,
    "Lleida": 31,
    "Lugo": 32,
    "Madrid": 33,
    "Málaga": 34,
    "Melilla": 35,
    "Murcia": 36,
    "Navarra": 37,
    "Ourense": 38,
    "Palencia": 39,
    "Pontevedra": 40,
    "Salamanca": 41,
    "Santa Cruz de Tenerife": 46,
    "Segovia": 42,
    "Sevilla": 43,
    "Soria": 44,
    "Tarragona": 45,
    "Teruel": 47,
    "Toledo": 48,
    "Valencia/València": 49,
    "Valladolid": 50,
    "Vizcaya/Bizkaia": 51,
    "Zamora": 52,
    "Zaragoza": 53
}

def generar_url_infojobs(puesto, modalidad=None, salario_minimo=None, localidades=None):
    """
    Genera una URL de búsqueda en InfoJobs para el puesto, modalidad, salario mínimo y localidades especificados.
    
    Parámetros:
        puesto (str): Texto a buscar (por ejemplo, 'Analista de datos').
        modalidad (str o list[str], opcional): Modalidad de trabajo. Puede ser un string o lista.
        salario_minimo (int, opcional): Salario mínimo anual bruto deseado (ej: 22000).
        localidades (str o list[str], opcional): Nombre de la(s) localidad(es) para filtrar.
    
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
    
    # Añadir salario mínimo si se especifica
    if salario_minimo is not None:
        params["salaryMin"] = salario_minimo
        params["salaryPeriod"] = "YEAR"
        params["salaryType"] = "GROSS"
    
    # Añadir localidades si se especifican
    if localidades:
        if isinstance(localidades, str):
            localidades = [localidades]
        # Buscar los códigos de las localidades y filtrar las que no existen
        codigos = [str(PROVINCIAS[loc]) for loc in localidades if loc in PROVINCIAS]
        if codigos:
            params["provinceIds"] = ",".join(codigos)
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"https://www.infojobs.net/ofertas-trabajo?{query_string}"