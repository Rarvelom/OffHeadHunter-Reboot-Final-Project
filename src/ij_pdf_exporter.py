from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import undetected_chromedriver as uc
import os
import base64
import time
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from src.job_offers import JobOfferStorage
import logging

# Cargar variables de entorno
load_dotenv()

# Configuración de MongoDB
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGODB_URI)
db = client['offheadhunter_db']

# Configuración de logging
logger = logging.getLogger(__name__)

def export_ij_offer_to_pdf(job_offer, output_pdf="oferta_infojobs.pdf"):
    """
    Descarga la página de una oferta de InfoJobs como PDF y la guarda en MongoDB.
    
    Parámetros:
        job_offer (dict): Objeto con los datos de la oferta de trabajo
        output_pdf (str): Nombre del archivo PDF (opcional, se usa para el nombre del campo)
    """
    offer_url = job_offer.get('url')
    if not offer_url:
        raise ValueError("La oferta debe contener una URL válida")
    
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless=new')  # Descomentar para producción
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1200x1600')

    driver = uc.Chrome(options=options)
    try:
        print(f"Accediendo a: {offer_url}")
        driver.get(offer_url)
        time.sleep(2)  # Espera a que cargue todo

        # Manejo de cookies
        try:
            disagree_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "didomi-notice-disagree-button"))
            )
            disagree_button.click()
            WebDriverWait(driver, 3).until(
                EC.invisibility_of_element_located((By.ID, "didomi-notice-disagree-button"))
            )
            print("Popup de cookies cerrado correctamente!")
        except TimeoutException:
            print("No se encontró popup de cookies o no se pudo cerrar (timeout).")
        except Exception as e:
            print(f"Error al manejar el popup de cookies: {e}")
        
        time.sleep(1)
        
        # Generar PDF
        pdf = driver.execute_cdp_cmd("Page.printToPDF", {
            "printBackground": True,
            "landscape": False,
            "paperWidth": 8.27,
            "paperHeight": 11.69,
        })
        
        # Obtener el binario del PDF
        pdf_binary = base64.b64decode(pdf['data'])
        
        # Construir documento para MongoDB
        job_document = {
            'external_id': job_offer.get('external_id'),
            'source_id': job_offer.get('source_id'),
            'title': job_offer.get('title', 'Sin título'),
            'company': job_offer.get('company', 'Empresa no especificada'),
            'locations': job_offer.get('locations', []),
            'description': job_offer.get('description', ''),
            'url': offer_url,
            'posted_at': job_offer.get('posted_at', datetime.utcnow()),
            'scraped_at': datetime.utcnow(),
            'tags': job_offer.get('tags', []),
            'salary_range': job_offer.get('salary_range', {
                'currency': 'EUR',
                'period': 'year'
            }),
            'is_active': True,
            'pdf_file': pdf_binary,
            'pdf_filename': output_pdf
        }
        
        # Insertar en MongoDB
        result = db.job_offers.insert_one(job_document)
        print(f"Oferta guardada en MongoDB con ID: {result.inserted_id}")
        
        # Guardar en Qdrant usando JobOfferStorage
        try:
            job_storage = JobOfferStorage(collection_name="job_embeddings_BGE")
            
            # Crear un diccionario solo con los campos necesarios para Qdrant
            qdrant_doc = {
                '_id': str(result.inserted_id),
                'title': job_offer.get('title', ''),
                'company': job_offer.get('company', ''),
                'description': job_offer.get('description', ''),
                'url': job_offer.get('url', ''),
                'locations': job_offer.get('locations', []),
                'salary_range': job_offer.get('salary_range', {
                    'currency': 'EUR',
                    'period': 'year'
                }),
                'posted_at': job_offer.get('posted_at', datetime.utcnow().isoformat()),
                'tags': job_offer.get('tags', []),
                'pdf_file': job_document.get('pdf_file')  # Pasamos el binario del PDF
            }
            
            embedding_info = job_storage.save_offer(qdrant_doc)
            
            if embedding_info:
                # Actualizar el documento en MongoDB con la información del embedding
                update_data = {
                    'embedding_vector_id_qdrant': embedding_info['embedding_vector_id_qdrant'],
                    'embedding_model': embedding_info['embedding_model']
                }
                
                db.job_offers.update_one(
                    {'_id': result.inserted_id},
                    {'$set': update_data}
                )
                logger.info(f"Documento de oferta actualizado en MongoDB con información de embedding")
            else:
                logger.warning("No se pudo guardar la oferta en Qdrant")
        except Exception as e:
            logger.error(f"Error al guardar la oferta en Qdrant: {e}")
        
        return {
            'success': True,
            'mongodb_id': str(result.inserted_id),
            'offer_title': job_document['title'],
            'company': job_document['company']
        }

    except Exception as e:
        print(f"Error al procesar la oferta: {e}")
        return {
            'success': False,
            'error': str(e),
            'offer_url': offer_url
        }

    finally:
        try:
            driver.quit()
        except:
            pass