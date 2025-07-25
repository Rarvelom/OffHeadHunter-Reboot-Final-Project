"""
Script para probar la conversión de timestamps entre CVs y ofertas de empleo.
"""
import sys
import os
from datetime import datetime, timezone

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.time_utils import to_unix_timestamp

def test_conversion():
    """Prueba la conversión de diferentes formatos de fecha."""
    print("=== PRUEBA DE CONVERSIÓN DE TIMESTAMPS ===\n")
    
    # 1. Timestamp actual
    now = datetime.now(timezone.utc)
    print(f"1. Ahora: {now.isoformat()}")
    print(f"   -> Timestamp: {to_unix_timestamp(now)}")
    
    # 2. String ISO con timezone
    iso_str = "2025-07-25T10:30:00+01:00"
    print(f"\n2. String ISO: {iso_str}")
    print(f"   -> Timestamp: {to_unix_timestamp(iso_str)}")
    
    # 3. Timestamp numérico
    ts = 1719304200  # 2025-07-25T09:30:00+00:00
    print(f"\n3. Timestamp numérico: {ts}")
    print(f"   -> Mismo timestamp: {to_unix_timestamp(ts) == ts}")

if __name__ == "__main__":
    test_conversion()
