import sys
from job_search_agent import JobSearchAgent # 1. Ejecutar el agente conversacional para obtener los criterios del usuario
from src.url_gen import generar_url_infojobs # 2. Generar la URL de búsqueda con url_gen.py
from src.ij_jobs_scraper import scrape_jobs # 3. Scraping de ofertas con jobs_scraper.py
from src.ij_pdf_exporter import export_ij_offer_to_pdf # 4. Exportar PDFs con pdf_scraper.py

def main():
    # Paso 1: Obtener criterios del usuario
    agent = JobSearchAgent()
    agent.run(reset_profile=True)
    user_profile = agent.user_profile  # Diccionario con los datos

    # Extraer los datos del perfil del usuario
    job_title = user_profile.get('job_title')
    work_modality = user_profile.get('work_modality')
    salary_expectation = user_profile.get('salary_expectation')
    location = user_profile.get('location')

    if not job_title:
        print("No se ha proporcionado un título de trabajo. Abortando.")
        sys.exit(1)

    # Paso 2: Generar URL de búsqueda con todos los parámetros
    url = generar_url_infojobs(
        puesto=job_title,
        modalidad=work_modality,
        salario_minimo=salary_expectation,
        localidades=location
    )
    print(f"URL generada para scraping: {url}")

    # Paso 3: Scraping de ofertas (guarda en MongoDB y devuelve la lista)
    job_offers = scrape_jobs(url)
    print(f"Ofertas extraídas: {len(job_offers)}")

    # Paso 4: Exportar PDFs de las ofertas
    for offer in job_offers:
        if offer.get("url"):
            export_ij_offer_to_pdf(offer)
        else:
            print("Oferta sin URL, se omite PDF.")

if __name__ == "__main__":
    main()