from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import undetected_chromedriver as uc
import time
import random

# Set up the Chrome driver
USER_AGENTS = [
    # A few example user agents
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

options = uc.ChromeOptions()
# Comment out headless for stealthier operation
# options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--disable-dev-shm-usage')
user_agent = random.choice(USER_AGENTS)
options.add_argument(f'user-agent={user_agent}')

# Optionally, add window size
options.add_argument('window-size=1200x800')

# Start undetected Chrome
driver = uc.Chrome(options=options)

# Got to the job page
page_url = input("Enter the job page URL: ")

# page_url = "https://www.infojobs.net/leganes/dependiente-cafeteria-20-hs-sem-cc-parquesur-sustitucion-leganes/of-i7489b0f9994dae83bb464d6e7c6fa1"

driver.get(page_url)

# Wait for and click the popup button if it appears
try:
    disagree_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "didomi-notice-disagree-button"))
    )
    disagree_button.click()
    # Wait for popup to disappear
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.ID, "didomi-notice-disagree-button"))
    )
except TimeoutException:
    print("No popup or could not click disagree button (timeout).")
except Exception as e:
    print("No popup or could not click disagree button:", e)

try:
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".ij-BaseTypography.ij-Heading.ij-Heading-title1"))
    )
except TimeoutException:
    print("Job title not found after waiting.")
    print(driver.page_source)  # Debug: print the HTML to see what's loaded
    driver.quit()
    exit(1)

page = driver.page_source
soup = BeautifulSoup(page, 'html.parser')

job_title = soup.select_one('h1.ij-BaseTypography.ij-Heading.ij-Heading-title1')
if job_title:
    print(job_title.text.strip())
else:
    print("Job title element not found in HTML.")

job_company = soup.select_one('a.ij-Link.ij-BaseTypography.ij-BaseTypography-primary.ij-Heading.ij-Heading-headline2')
if job_company:
    print(job_company.text.strip())
else:
    print("Job company element not found in HTML.")

details_container = soup.select_one('div.ij-Box.ij-OfferDetailHeader-details')
if details_container:
    detail_ps = details_container.select('p.ij-BaseTypography.ij-Text.ij-Text-body1')
    location_city = location_province = modality = salary = experience = contract_type = None

    for idx, p in enumerate(detail_ps):
        text = p.get_text(strip=True)
        if idx == 0:
            # Location (has an <a> child)
            if p.find('a'):
                location_city = p.contents[0].strip().replace('(', '').strip()
                location_province = p.find('a').text.strip()
        elif idx == 1:
            modality = text
        elif idx == 2:
            salary = text
        elif idx == 3:
            experience = text
        elif idx == 4:
            contract_type = text

    print(f"Job location: {location_city}, {location_province}")
    print(f"Job modality: {modality}")
    print(f"Job salary: {salary}")
    print(f"Minimum experience: {experience}")
    print(f"Contract type: {contract_type}")
else:
    print("Job details container not found in HTML.")

# Extract publication date
published_ul = soup.select_one('ul.ij-Box.ij-OfferDetailHeader-publishedAt')
if published_ul:
    published_li = published_ul.select_one('li.ij-BaseTypography.ij-BaseTypography-gray.ij-Text.ij-Text-caption')
    if published_li:
        published_text = ' '.join(published_li.stripped_strings)
        print(f"Published: {published_text}")
    else:
        print("Publication date <li> not found in HTML.")
else:
    print("Publication date <ul> not found in HTML.")

# Give time to view output before closing browser
time.sleep(5)
driver.quit()

### NOTE ### LAUNCH ME USING: 
# "xvfb-run -a python3 src/scraper.py"

### NOTE 2 ###
# Para funcionar adecuadamente, se debe usar una version de Chrome >= 138