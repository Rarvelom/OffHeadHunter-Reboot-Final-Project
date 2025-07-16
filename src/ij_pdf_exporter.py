from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import undetected_chromedriver as uc
import os
import base64
import time

def export_ij_offer_to_pdf(offer_url, output_pdf="oferta_infojobs.pdf"):
    """
    Descarga la página de una oferta de InfoJobs como PDF usando Chrome headless.
    Parámetros:
        offer_url (str): URL de la oferta de InfoJobs.
        output_pdf (str): Nombre del archivo PDF de salida.
    """
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1200x1600')

    driver = uc.Chrome(options=options)
    try:
        driver.get(offer_url)
        time.sleep(1)  # Espera a que cargue todo

        try:
            disagree_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.ID, "didomi-notice-disagree-button"))
            )
            disagree_button.click()
            # Espera a que el popup desaparezca (Dejar el parametro timeout SIEMPRE a 5! Menos tiempo no funciona)
            WebDriverWait(driver, 3).until(
                EC.invisibility_of_element_located((By.ID, "didomi-notice-disagree-button"))
            )
            print("¡Popup de cookies cerrado correctamente!")
        except TimeoutException:
            print("No popup o no se pudo cerrar el popup (timeout).")
        except Exception as e:
            print("No popup o no se pudo cerrar el popup:", e)
        
        time.sleep(1)
        
        pdf = driver.execute_cdp_cmd("Page.printToPDF", {
            "printBackground": True,
            "landscape": False,
            "paperWidth": 8.27,
            "paperHeight": 11.69,
        })
        
        # Asegúrate de que la carpeta pdf/ existe
        os.makedirs("pdf", exist_ok=True)
        pdf_path = os.path.join("pdf", output_pdf)
        # Guardar el PDF generado en la carpeta pdf/
        with open(pdf_path, "wb") as f:
            f.write(base64.b64decode(pdf['data']))
        print(f"PDF guardado como {pdf_path}")
    
    except Exception as e:
        print(f"Error al guardar el PDF: {e}")

    finally:
        time.sleep(1)
        driver.quit()

# Ejemplo de uso:
export_ij_offer_to_pdf("https://www.infojobs.net/belmonte-de-miranda/tecnico-prl/of-ib3308c330d429791e381a092ed88ad")