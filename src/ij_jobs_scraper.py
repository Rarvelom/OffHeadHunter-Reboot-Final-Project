from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from datetime import datetime, timedelta, timezone
import undetected_chromedriver as uc
import time
import random
import json
import re

# Configuración del driver de Chrome no detectado
USER_AGENTS = [
    # Algunos user agents de ejemplo
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

options = uc.ChromeOptions()
# Descomenta la siguiente línea si quieres modo headless para mayor sigilo
# options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--disable-dev-shm-usage')
user_agent = random.choice(USER_AGENTS)
options.add_argument('--window-size=1200x800')
options.add_argument(f'--user-agent={user_agent}')

# Opcionalmente, establece el tamaño de la ventana

# Inicia Chrome no detectado
driver = uc.Chrome(options=options)

# URL de la página de ofertas
page_url = "https://www.infojobs.net/ofertas-trabajo"

# Si deseas usar una URL personalizada, descomenta la siguiente línea y comenta la anterior
# page_url = input("Enter the job page URL: ")

driver.get(page_url)

# Espera y haz clic en el botón de cookies si aparece
try:
    disagree_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "didomi-notice-disagree-button"))
    )
    disagree_button.click()
    # Espera a que el popup desaparezca (Dejar el parametro timeout SIEMPRE a 5! Menos tiempo no funciona)
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.ID, "didomi-notice-disagree-button"))
    )
    print("¡Popup de cookies cerrado correctamente!")
except TimeoutException:
    print("No popup o no se pudo cerrar el popup (timeout).")
except Exception as e:
    print("No popup o no se pudo cerrar el popup:", e)

# Scroll humano y simulación de movimiento de ratón para cargar todas las ofertas
SCROLL_PAUSE_TIME = 0.5
SCROLL_STEP = 500
MAX_SCROLL = 5500  # Ajusta según la longitud estimada de la página

actions = ActionChains(driver)
for y in range(0, MAX_SCROLL, SCROLL_STEP):
    driver.execute_script(f"window.scrollTo(0, {y});")
    actions.move_by_offset(0, 10).perform()  # Simula movimiento del ratón
    time.sleep(SCROLL_PAUSE_TIME)

page = driver.page_source
soup = BeautifulSoup(page, 'html.parser')

# Encuentra todas las tarjetas de ofertas con el selector más específico
offer_cards = soup.select(
    'li.ij-List-item.sui-PrimitiveLinkBox > div.sui-AtomCard.sui-AtomCard-link.sui-AtomCard--rounded-l > div.sui-AtomCard-info'
)

# ID de fuente de ejemplo para pruebas (actualiza según tu entorno)
SOURCE_ID = "64e5c2d6f0a5e7a4f3b1c2d3"

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
            return (now - timedelta(minutes=mins)).isoformat()
        elif 'h' in fecha:  # horas
            hours = int(fecha.replace('h', '').strip())
            return (now - timedelta(hours=hours)).isoformat()
        elif 'día' in fecha or 'días' in fecha:  # días
            days = int(re.findall(r'(\d+)', fecha)[0])
            return (now - timedelta(days=days)).isoformat()
    except Exception:
        return None
    return None

results = []
for card in offer_cards:
    # URL y título de la oferta
    title_a = card.select_one('a.ij-OfferCardContent-description-title-link')
    offer_url = title_a['href'] if title_a and title_a.has_attr('href') else None
    if offer_url and offer_url.startswith('//'):
        offer_url = 'https:' + offer_url
    title = title_a.text.strip() if title_a else None

    # Empresa
    company_a = card.select_one('a.ij-OfferCardContent-description-subtitle-link')
    company = company_a.text.strip() if company_a else None

    # Localización
    location_span = card.select_one('span.ij-OfferCardContent-description-list-item-truncate')
    location = location_span.text.strip() if location_span else None
    locations = [location] if location else []

    # Modalidad (segundo <li> de la primera ul)
    modality = None
    ul_lists = card.select('ul.ij-OfferCardContent-description-list')
    if ul_lists and len(ul_lists) > 0:
        li_items = ul_lists[0].select('li.ij-OfferCardContent-description-list-item')
        if len(li_items) > 1:
            modality = li_items[1].text.strip()

    # Fecha de publicación
    published_span = card.select_one('span.ij-OfferCardContent-description-published')
    published = published_span.text.strip() if published_span else None
    posted_at = parse_posted_at(published)

    # Descripción
    desc_p = card.select_one('p.ij-OfferCardContent-description-description')
    description = desc_p.text.strip() if desc_p else None

    # Contrato y jornada (primer y segundo <li> de la segunda ul)
    contract_type = workday_type = None
    if ul_lists and len(ul_lists) > 1:
        li_items2 = ul_lists[1].select('li.ij-OfferCardContent-description-list-item')
        hide_on_mobile_items = [li for li in li_items2 if 'ij-OfferCardContent-description-list-item--hideOnMobile' in li.get('class', [])]
        if len(hide_on_mobile_items) > 0:
            contract_type = hide_on_mobile_items[0].text.strip()
        if len(hide_on_mobile_items) > 1:
            workday_type = hide_on_mobile_items[1].text.strip()

    # Salario
    salary = None
    salary_info = card.select_one('span.ij-OfferCardContent-description-salary-info')
    salary_no_info = card.select_one('span.ij-OfferCardContent-description-salary-no-information')
    if salary_info:
        salary = salary_info.text.strip()
    elif salary_no_info:
        salary = salary_no_info.text.strip()
    salary_range = parse_salary(salary)

    # Tags
    tags = list(filter(None, [contract_type, workday_type, modality]))

    # external_id
    external_id = extract_external_id(offer_url) if offer_url else None

    # scraped_at
    scraped_at = datetime.now(timezone.utc).isoformat()

    mongo_job = {
        "external_id": external_id,
        "source_id": SOURCE_ID,
        "title": title,
        "company": company,
        "locations": locations,
        "description": description,
        "url": offer_url,
        "posted_at": posted_at,
        "scraped_at": scraped_at,
        "tags": tags,
        "salary_range": salary_range,
        "is_active": True,
        "expires_at": None
    }
    results.append(mongo_job)

with open("src/jsons/jobs_ij.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Se han guardado {len(results)} ofertas en 'jobs_ij.json'")

time.sleep(2)

driver.quit()