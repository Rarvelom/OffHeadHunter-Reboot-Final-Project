"""
Utility functions for consistent timestamp handling across the application.
"""
from datetime import datetime, timezone
from typing import Union, Optional


def get_current_utc_timestamp() -> int:
    """Get current UTC time as Unix timestamp (seconds since epoch)."""
    return int(datetime.now(timezone.utc).timestamp())


def to_unix_timestamp(dt: Union[datetime, str, int, float, None]) -> Optional[int]:
    """
    Convert various datetime formats to Unix timestamp (seconds since epoch).
    
    Args:
        dt: Input datetime (datetime object, ISO format string, or Unix timestamp)
        
    Returns:
        Unix timestamp in seconds, or None if input is None or invalid
    """
    if dt is None:
        return None
        
    try:
        # If already a Unix timestamp (int or float)
        if isinstance(dt, (int, float)):
            return int(dt)
            
        # If string, parse it
        if isinstance(dt, str):
            # Handle ISO format with or without timezone
            if 'T' in dt:
                if dt.upper().endswith('Z'):
                    dt = dt[:-1] + '+00:00'  # Convert Z to +00:00 for fromisoformat
                dt_obj = datetime.fromisoformat(dt)
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                return int(dt_obj.timestamp())
            # Try to parse as Unix timestamp string
            try:
                return int(float(dt))
            except (ValueError, TypeError):
                pass
        
        # If datetime object
        if hasattr(dt, 'timestamp'):
            return int(dt.timestamp())
            
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert {dt} to timestamp: {e}")
        
    return None


def to_iso_format(timestamp: Union[datetime, int, float, str, None]) -> Optional[str]:
    """
    Convert various datetime formats to ISO 8601 format string.
    
    Args:
        timestamp: Input datetime (datetime object, Unix timestamp, or ISO string)
        
    Returns:
        ISO 8601 formatted string, or None if input is invalid
    """
    if timestamp is None:
        return None
        
    try:
        # If already a string, validate it's ISO format
        if isinstance(timestamp, str):
            # Try to parse and re-format to ensure it's valid ISO
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.isoformat()
            
        # If Unix timestamp (int or float)
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, timezone.utc)
            return dt.isoformat()
            
        # If datetime object
        if hasattr(timestamp, 'isoformat'):
            dt = timestamp
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
            
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert {timestamp} to ISO format: {e}")
        
    return None


def is_after(
    timestamp1: Union[datetime, int, float, str, None],
    timestamp2: Union[datetime, int, float, str, None]
) -> Optional[bool]:
    """
    Check if timestamp1 is after timestamp2.
    
    Args:
        timestamp1: First timestamp to compare
        timestamp2: Second timestamp to compare against
        
    Returns:
        True if timestamp1 > timestamp2, False if timestamp1 <= timestamp2,
        or None if either timestamp is invalid
    """
    ts1 = to_unix_timestamp(timestamp1)
    ts2 = to_unix_timestamp(timestamp2)
    
    if ts1 is None or ts2 is None:
        return None
        
    return ts1 > ts2
