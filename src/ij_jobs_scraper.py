import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from src.utils.time_utils import get_current_utc_timestamp, to_iso_format, to_unix_timestamp
import random
import time
import re
import os
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv

# --- Funciones auxiliares extract_external_id, parse_salary, parse_posted_at ---
def extract_external_id(url):
    match = re.search(r'/of-([a-zA-Z0-9]+)', url)
    return match.group(1) if match else None

def parse_salary(salary_str):
    if not salary_str or "no disponible" in salary_str.lower():
        return None
    match = re.findall(r'([0-9]+)[\s]*€', salary_str.replace('.', ''))
    if match:
        min_salary = int(match[0])
        max_salary = int(match[1]) if len(match) > 1 else min_salary
        return {"min": min_salary, "max": max_salary, "currency": "EUR"}
    return None

def parse_posted_at(fecha_publicacion):
    now = datetime.now(timezone.utc)
    if not fecha_publicacion:
        return None
    try:
        fecha = fecha_publicacion.lower().replace('hace', '').strip()
        if 'm' in fecha:  # minutos
            mins = int(fecha.replace('m', '').strip())
            dt = now - timedelta(minutes=mins)
        elif 'h' in fecha:  # horas
            hours = int(fecha.replace('h', '').strip())
            dt = now - timedelta(hours=hours)
        elif 'día' in fecha or 'días' in fecha:  # días
            days = int(re.findall(r'(\d+)', fecha)[0])
            dt = now - timedelta(days=days)
        else:
            return None
            
        # Convertir a timestamp Unix (segundos desde epoch)
        return int(dt.timestamp())
    except Exception as e:
        print(f"Error al parsear fecha '{fecha_publicacion}': {e}")
        return None

USER_AGENTS = [
    # Algunos user agents de ejemplo
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

SOURCE_ID = "64e5c2d6f0a5e7a4f3b1c2d3"  # O el que uses

# !!!BORRAR MAS ADELANTE!!! FUNCIÓN CON LÓGICA DUPLICADA EN "ij_pdf_exporter.py", de momento se ha desactivado su inicialización en "def scrape_jobs()" para evitar generar entradas duplicadas.
def save_to_mongodb(jobs_data):
    load_dotenv()
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        print("❌ MONGODB_URI no configurada en el entorno.")
        return
    client = MongoClient(mongodb_uri)
    db = client['offheadhunter_db']
    collection = db['job_offers']
    success_count = 0
    for job in jobs_data:
        try:
            if job.get('external_id'):
                job['_id'] = job['external_id']
                result = collection.update_one(
                    {'_id': job['_id']},
                    {'$set': job},
                    upsert=True
                )
                success_count += 1
        except DuplicateKeyError:
            print(f"⚠️  Duplicado: {job.get('external_id')}")
        except Exception as e:
            print(f"❌ Error guardando {job.get('external_id')}: {e}")
    print(f"✅ Guardadas/actualizadas {success_count} ofertas en MongoDB.")
    client.close()

def scrape_jobs(page_url):
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1200x800')
    user_agent = random.choice(USER_AGENTS)
    options.add_argument(f'--user-agent={user_agent}')
    driver = uc.Chrome(options=options)

    try:
        driver.get(page_url)
        # Manejo popup cookies
        try:
            disagree_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "didomi-notice-disagree-button"))
            )
            disagree_button.click()
            WebDriverWait(driver, 10).until(
                EC.invisibility_of_element_located((By.ID, "didomi-notice-disagree-button"))
            )
        except TimeoutException:
            pass
        except Exception:
            pass

        # Hacer click en el botón con id="buttonKeyword" tras el manejo de cookies
        try:
            keyword_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "buttonKeyword"))
            )
            keyword_button.click()
            time.sleep(3)
        except Exception as e:
            print(f"No se pudo hacer click en el botón 'buttonKeyword': {e}")

        # Scroll humano y simulación de movimiento de ratón para cargar todas las ofertas
        SCROLL_PAUSE_TIME = 0.5
        SCROLL_STEP = 500
        MAX_SCROLL = 5500
        actions = ActionChains(driver)
        for y in range(0, MAX_SCROLL, SCROLL_STEP):
            driver.execute_script(f"window.scrollTo(0, {y});")
            actions.move_by_offset(0, 10).perform()
            time.sleep(SCROLL_PAUSE_TIME)

        page = driver.page_source
        soup = BeautifulSoup(page, 'html.parser')
        offer_cards = soup.select(
            'li.ij-List-item.sui-PrimitiveLinkBox > div.sui-AtomCard.sui-AtomCard-link.sui-AtomCard--rounded-l > div.sui-AtomCard-info'
        )

        results = []
        # for card in offer_cards: # DESCOMENTAR PARA PRODUCCIÓN
        for card in offer_cards:  # SOLO PARA PRUEBAS!!! procesa los primeros 5 elementos
            title_a = card.select_one('a.ij-OfferCardContent-description-title-link')
            offer_url = title_a['href'] if title_a and title_a.has_attr('href') else None
            if offer_url and offer_url.startswith('//'):
                offer_url = 'https:' + offer_url
            title = title_a.text.strip() if title_a else None

            company_a = card.select_one('a.ij-OfferCardContent-description-subtitle-link')
            company = company_a.text.strip() if company_a else None

            location_span = card.select_one('span.ij-OfferCardContent-description-list-item-truncate')
            location = location_span.text.strip() if location_span else None
            locations = [location] if location else []

            # Modalidad
            modality = None
            ul_lists = card.select('ul.ij-OfferCardContent-description-list')
            if ul_lists and len(ul_lists) > 0:
                li_items = ul_lists[0].select('li.ij-OfferCardContent-description-list-item')
                if len(li_items) > 1:
                    modality = li_items[1].text.strip()

            published_span = card.select_one('span.ij-OfferCardContent-description-published')
            published = published_span.text.strip() if published_span else None
            posted_at = parse_posted_at(published)

            desc_p = card.select_one('p.ij-OfferCardContent-description-description')
            description = desc_p.text.strip() if desc_p else None

            contract_type = workday_type = None
            if ul_lists and len(ul_lists) > 1:
                li_items2 = ul_lists[1].select('li.ij-OfferCardContent-description-list-item')
                hide_on_mobile_items = [li for li in li_items2 if 'ij-OfferCardContent-description-list-item--hideOnMobile' in li.get('class', [])]
                if len(hide_on_mobile_items) > 0:
                    contract_type = hide_on_mobile_items[0].text.strip()
                if len(hide_on_mobile_items) > 1:
                    workday_type = hide_on_mobile_items[1].text.strip()

            salary = None
            salary_info = card.select_one('span.ij-OfferCardContent-description-salary-info')
            salary_no_info = card.select_one('span.ij-OfferCardContent-description-salary-no-information')
            if salary_info:
                salary = salary_info.text.strip()
            elif salary_no_info:
                salary = salary_no_info.text.strip()
            salary_range = parse_salary(salary)

            tags = list(filter(None, [contract_type, workday_type, modality]))
            external_id = extract_external_id(offer_url) if offer_url else None
            scraped_at = get_current_utc_timestamp()

            mongo_job = {
                "external_id": external_id,
                "source_id": SOURCE_ID,
                "title": title,
                "company": company,
                "locations": locations,
                "description": description,
                "url": offer_url,
                "posted_at": to_unix_timestamp(posted_at) if posted_at else None,
                "scraped_at": scraped_at,
                "tags": tags,
                "salary_range": salary_range,
                "is_active": True,
                "expires_at": None
            }
            results.append(mongo_job)

        # save_to_mongodb(results)
        
        return results

    finally:
        driver.quit()

###########################################
##########################################

# # Para pruebas manuales:
# if __name__ == "__main__":
#     # url = input("Introduce la URL de búsqueda de InfoJobs: ")
#     url = "https://www.infojobs.net/ofertas-trabajo?keyword=Analista%20de%20datos&segmentId=&page=1&sortBy=PUBLICATION_DATE&onlyForeignCountry=false&sinceDate=ANY&teleworkingIds=1&salaryMin=30000&salaryPeriod=YEAR&salaryType=GROSS&provinceIds=33"
#     ofertas = scrape_jobs(url)
#     print(f"Ofertas extraídas: {len(ofertas)}")
#     # Puedes guardar en JSON aquí si lo necesitas para debug

#############################################
#############################################

# def scrape_testing(page_url):
#     options = uc.ChromeOptions()
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-blink-features=AutomationControlled')
#     options.add_argument('--disable-dev-shm-usage')
#     options.add_argument('--window-size=1200x800')
#     user_agent = random.choice(USER_AGENTS)
#     options.add_argument(f'--user-agent={user_agent}')
#     driver = uc.Chrome(options=options)

#     try:
#         driver.get(page_url)
#         # Manejo popup cookies
#         try:
#             disagree_button = WebDriverWait(driver, 10).until(
#                 EC.element_to_be_clickable((By.ID, "didomi-notice-disagree-button"))
#             )
#             disagree_button.click()
#             WebDriverWait(driver, 10).until(
#                 EC.invisibility_of_element_located((By.ID, "didomi-notice-disagree-button"))
#             )
#         except TimeoutException:
#             pass
#         except Exception:
#             pass

#         # Hacer click en el botón con id="buttonKeyword" tras el manejo de cookies
#         try:
#             keyword_button = WebDriverWait(driver, 10).until(
#                 EC.element_to_be_clickable((By.ID, "buttonKeyword"))
#             )
#             keyword_button.click()
#             import time
#             time.sleep(3)
#         except Exception as e:
#             print(f"No se pudo hacer click en el botón 'buttonKeyword': {e}")

#         # Scroll humano y simulación de movimiento de ratón para cargar todas las ofertas
#         SCROLL_PAUSE_TIME = 0.5
#         SCROLL_STEP = 500
#         MAX_SCROLL = 5500
#         actions = ActionChains(driver)
#         for y in range(0, MAX_SCROLL, SCROLL_STEP):
#             driver.execute_script(f"window.scrollTo(0, {y});")
#             actions.move_by_offset(0, 10).perform()
#             time.sleep(SCROLL_PAUSE_TIME)

#         page = driver.page_source
#         soup = BeautifulSoup(page, 'html.parser')
#         offer_cards = soup.select(
#             'li.ij-List-item.sui-PrimitiveLinkBox > div.sui-AtomCard.sui-AtomCard-link.sui-AtomCard--rounded-l > div.sui-AtomCard-info'
#         )

#         results = []
#         for card in offer_cards: # DESCOMENTAR PARA PRODUCCIÓN
#         # for card in offer_cards[:2]:  # SOLO PARA PRUEBAS!!! procesa los primeros 5 elementos
#             title_a = card.select_one('a.ij-OfferCardContent-description-title-link')
#             offer_url = title_a['href'] if title_a and title_a.has_attr('href') else None
#             if offer_url and offer_url.startswith('//'):
#                 offer_url = 'https:' + offer_url
#             title = title_a.text.strip() if title_a else None

#             company_a = card.select_one('a.ij-OfferCardContent-description-subtitle-link')
#             company = company_a.text.strip() if company_a else None

#             location_span = card.select_one('span.ij-OfferCardContent-description-list-item-truncate')
#             location = location_span.text.strip() if location_span else None
#             locations = [location] if location else []

#             # Modalidad
#             modality = None
#             ul_lists = card.select('ul.ij-OfferCardContent-description-list')
#             if ul_lists and len(ul_lists) > 0:
#                 li_items = ul_lists[0].select('li.ij-OfferCardContent-description-list-item')
#                 if len(li_items) > 1:
#                     modality = li_items[1].text.strip()

#             published_span = card.select_one('span.ij-OfferCardContent-description-published')
#             published = published_span.text.strip() if published_span else None
#             posted_at = parse_posted_at(published)

#             desc_p = card.select_one('p.ij-OfferCardContent-description-description')
#             description = desc_p.text.strip() if desc_p else None

#             contract_type = workday_type = None
#             if ul_lists and len(ul_lists) > 1:
#                 li_items2 = ul_lists[1].select('li.ij-OfferCardContent-description-list-item')
#                 hide_on_mobile_items = [li for li in li_items2 if 'ij-OfferCardContent-description-list-item--hideOnMobile' in li.get('class', [])]
#                 if len(hide_on_mobile_items) > 0:
#                     contract_type = hide_on_mobile_items[0].text.strip()
#                 if len(hide_on_mobile_items) > 1:
#                     workday_type = hide_on_mobile_items[1].text.strip()

#             salary = None
#             salary_info = card.select_one('span.ij-OfferCardContent-description-salary-info')
#             salary_no_info = card.select_one('span.ij-OfferCardContent-description-salary-no-information')
#             if salary_info:
#                 salary = salary_info.text.strip()
#             elif salary_no_info:
#                 salary = salary_no_info.text.strip()
#             salary_range = parse_salary(salary)

#             tags = list(filter(None, [contract_type, workday_type, modality]))
#             external_id = extract_external_id(offer_url) if offer_url else None
#             scraped_at = datetime.now(timezone.utc).isoformat()

#             mongo_job = {
#                 "external_id": external_id,
#                 "source_id": SOURCE_ID,
#                 "title": title,
#                 "company": company,
#                 "locations": locations,
#                 "description": description,
#                 "url": offer_url,
#                 "posted_at": posted_at,
#                 "scraped_at": scraped_at,
#                 "tags": tags,
#                 "salary_range": salary_range,
#                 "is_active": True,
#                 "expires_at": None
#             }
#             results.append(mongo_job)
        
#         # Guardar los resultados en un archivo JSON antes de retornar
#         import json
#         with open("testing.json", "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
        
#         return results

#     finally:
#         driver.quit()

# url = "https://www.infojobs.net/ofertas-trabajo?keyword=Analista%20de%20datos&segmentId=&page=1&sortBy=RELEVANCE&onlyForeignCountry=false&sinceDate=ANY&teleworkingIds=1&salaryMin=30000&salaryPeriod=YEAR&salaryType=GROSS"

# scrape_testing(url)

# # print(test_result)