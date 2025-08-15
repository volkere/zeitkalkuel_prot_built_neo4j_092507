
from __future__ import annotations
from typing import Optional, Dict, Any
from PIL import Image, ExifTags
from datetime import datetime
import piexif

def extract_exif_gps(image_path: str) -> Optional[Dict[str, float]]:
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        exif_data = { ExifTags.TAGS.get(k,k): v for k,v in exif.items() }
        gps_info = exif_data.get('GPSInfo')
        if not gps_info:
            return None
        gps_parsed = { ExifTags.GPSTAGS.get(t, t): gps_info[t] for t in gps_info }
        from .utils import dms_to_dd
        lat = dms_to_dd(gps_parsed.get('GPSLatitude'), gps_parsed.get('GPSLatitudeRef'))
        lon = dms_to_dd(gps_parsed.get('GPSLongitude'), gps_parsed.get('GPSLongitudeRef'))
        if lat is None or lon is None:
            return None
        return {'lat': lat, 'lon': lon}
    except Exception:
        return None

def extract_comprehensive_metadata(image_path: str) -> Dict[str, Any]:
    """Extrahierte umfassende Metadaten aus einem Bild"""
    metadata = {}
    
    try:
        # EXIF mit PIL
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            exif_data = { ExifTags.TAGS.get(k,k): v for k,v in exif.items() }
            
            # Datum und Zeit
            date_time = exif_data.get('DateTime')
            if date_time:
                try:
                    metadata['datetime'] = datetime.strptime(date_time, '%Y:%m:%d %H:%M:%S').isoformat()
                except:
                    metadata['datetime'] = date_time
            
            # Kamera-Informationen
            metadata['camera_make'] = exif_data.get('Make')
            metadata['camera_model'] = exif_data.get('Model')
            metadata['lens'] = exif_data.get('LensModel')
            
            # Aufnahme-Einstellungen
            metadata['focal_length'] = exif_data.get('FocalLength')
            metadata['f_number'] = exif_data.get('FNumber')
            metadata['iso'] = exif_data.get('ISOSpeedRatings')
            metadata['exposure_time'] = exif_data.get('ExposureTime')
            metadata['flash'] = exif_data.get('Flash')
            
            # Bildgröße und Format
            metadata['image_width'] = exif_data.get('ExifImageWidth') or img.width
            metadata['image_height'] = exif_data.get('ExifImageHeight') or img.height
            metadata['orientation'] = exif_data.get('Orientation')
            
            # GPS-Informationen
            gps_info = exif_data.get('GPSInfo')
            if gps_info:
                gps_parsed = { ExifTags.GPSTAGS.get(t, t): gps_info[t] for t in gps_info }
                from .utils import dms_to_dd
                lat = dms_to_dd(gps_parsed.get('GPSLatitude'), gps_parsed.get('GPSLatitudeRef'))
                lon = dms_to_dd(gps_parsed.get('GPSLongitude'), gps_parsed.get('GPSLongitudeRef'))
                if lat is not None and lon is not None:
                    metadata['gps'] = {'lat': lat, 'lon': lon}
                    
                    # Höhe
                    altitude = gps_parsed.get('GPSAltitude')
                    if altitude:
                        metadata['gps']['altitude'] = altitude[0] / altitude[1]
                    
                    # GPS-Zeitstempel
                    gps_time = gps_parsed.get('GPSTimeStamp')
                    gps_date = gps_parsed.get('GPSDateStamp')
                    if gps_time and gps_date:
                        try:
                            hour, minute, second = [t[0]/t[1] for t in gps_time]
                            metadata['gps']['timestamp'] = f"{gps_date} {int(hour):02d}:{int(minute):02d}:{int(second):02d}"
                        except:
                            pass
        
        # Zusätzliche EXIF-Daten mit piexif
        try:
            exif_dict = piexif.load(image_path)
            if '0th' in exif_dict:
                # Software
                if piexif.ImageIFD.Software in exif_dict['0th']:
                    metadata['software'] = exif_dict['0th'][piexif.ImageIFD.Software].decode('utf-8', errors='ignore')
                
                # Copyright
                if piexif.ImageIFD.Copyright in exif_dict['0th']:
                    metadata['copyright'] = exif_dict['0th'][piexif.ImageIFD.Copyright].decode('utf-8', errors='ignore')
                
                # Artist
                if piexif.ImageIFD.Artist in exif_dict['0th']:
                    metadata['artist'] = exif_dict['0th'][piexif.ImageIFD.Artist].decode('utf-8', errors='ignore')
            
            if 'Exif' in exif_dict:
                # Weißabgleich
                if piexif.ExifIFD.WhiteBalance in exif_dict['Exif']:
                    metadata['white_balance'] = exif_dict['Exif'][piexif.ExifIFD.WhiteBalance]
                
                # Belichtungsmodus
                if piexif.ExifIFD.ExposureMode in exif_dict['Exif']:
                    metadata['exposure_mode'] = exif_dict['Exif'][piexif.ExifIFD.ExposureMode]
                
                # Messmodus
                if piexif.ExifIFD.MeteringMode in exif_dict['Exif']:
                    metadata['metering_mode'] = exif_dict['Exif'][piexif.ExifIFD.MeteringMode]
        except:
            pass
            
    except Exception as e:
        metadata['error'] = str(e)
    
    return metadata

def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="photo_metadata_app")
        location = geolocator.reverse((lat, lon), timeout=10, exactly_one=True, language="de")
        if location:
            return location.address
        return None
    except Exception:
        return None

def get_location_details(lat: float, lon: float) -> Dict[str, Any]:
    """Erweiterte Standort-Informationen"""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        
        geolocator = Nominatim(user_agent="photo_metadata_app")
        
        # Basis-Adresse
        location = geolocator.reverse((lat, lon), timeout=10, exactly_one=True, language="de")
        
        if not location:
            return {}
        
        # Erweiterte Informationen
        address = location.raw.get('address', {})
        
        return {
            'full_address': location.address,
            'country': address.get('country'),
            'state': address.get('state'),
            'city': address.get('city') or address.get('town') or address.get('village'),
            'postcode': address.get('postcode'),
            'street': address.get('road'),
            'house_number': address.get('house_number'),
            'neighbourhood': address.get('neighbourhood'),
            'suburb': address.get('suburb'),
            'coordinates': {'lat': lat, 'lon': lon}
        }
        
    except Exception as e:
        return {'error': str(e)}
