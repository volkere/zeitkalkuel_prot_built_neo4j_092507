
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Tuple
import cv2
from datetime import datetime

def dms_to_dd(dms, ref) -> Optional[float]:
    if not dms or not ref:
        return None
    try:
        deg = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
        dd = deg + minutes/60.0 + seconds/3600.0
        if ref in ['S','W']:
            dd = -dd
        return dd
    except Exception:
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def assess_image_quality(image: np.ndarray) -> Dict[str, float]:
    """Bewertet die allgemeine Bildqualität"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Schärfe (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)
        
        # Helligkeit
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        # Kontrast
        contrast = np.std(gray)
        contrast_score = min(contrast / 50, 1.0)
        
        # Rauschbewertung (einfache Implementierung)
        noise_score = 1.0 - min(laplacian_var / 1000, 1.0)  # Niedrige Varianz = weniger Rauschen
        
        # Gesamtqualität
        overall_quality = (sharpness_score * 0.3 + brightness_score * 0.25 + 
                          contrast_score * 0.25 + noise_score * 0.2)
        
        return {
            'overall_quality': float(max(0.0, min(1.0, overall_quality))),
            'sharpness': float(sharpness_score),
            'brightness': float(brightness_score),
            'contrast': float(contrast_score),
            'noise_level': float(noise_score)
        }
    except Exception:
        return {
            'overall_quality': 0.5,
            'sharpness': 0.5,
            'brightness': 0.5,
            'contrast': 0.5,
            'noise_level': 0.5
        }

def extract_color_histogram(image: np.ndarray, bins: int = 32) -> Dict[str, np.ndarray]:
    """Extrahiert Farbhistogramme für Bildanalyse"""
    try:
        histograms = {}
        
        # BGR Histogramme
        for i, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            histograms[color] = hist.flatten()
        
        # Graustufen Histogramm
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        histograms['gray'] = gray_hist.flatten()
        
        # Normalisierte Histogramme
        histograms_normalized = {}
        for color, hist in histograms.items():
            histograms_normalized[f"{color}_normalized"] = hist / (np.sum(hist) + 1e-8)
        
        histograms.update(histograms_normalized)
        return histograms
        
    except Exception:
        return {}

def analyze_image_composition(image: np.ndarray) -> Dict[str, Any]:
    """Analysiert die Bildkomposition"""
    try:
        height, width = image.shape[:2]
        
        # Seitenverhältnis
        aspect_ratio = width / height
        
        # Bildgröße
        total_pixels = width * height
        
        # Zentrum des Bildes
        center_x, center_y = width // 2, height // 2
        
        # Dominante Farben (einfache Implementierung)
        # Konvertiere zu HSV für bessere Farbanalyse
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Durchschnittliche H, S, V Werte
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Farbtemperatur (einfache Schätzung)
        if avg_hue < 30 or avg_hue > 150:  # Blau/Cyan Bereich
            color_temperature = "cool"
        elif avg_hue > 30 and avg_hue < 90:  # Grün Bereich
            color_temperature = "neutral"
        else:  # Rot/Gelb Bereich
            color_temperature = "warm"
        
        return {
            'dimensions': {'width': width, 'height': height},
            'aspect_ratio': float(aspect_ratio),
            'total_pixels': int(total_pixels),
            'center': {'x': center_x, 'y': center_y},
            'avg_hue': float(avg_hue),
            'avg_saturation': float(avg_saturation),
            'avg_value': float(avg_value),
            'color_temperature': color_temperature
        }
        
    except Exception:
        return {}

def parse_datetime_string(datetime_str: str) -> Optional[datetime]:
    """Parst verschiedene Datetime-Formate"""
    formats = [
        '%Y:%m:%d %H:%M:%S',  # EXIF Standard
        '%Y-%m-%d %H:%M:%S',  # ISO Format
        '%Y-%m-%dT%H:%M:%S',  # ISO mit T
        '%Y-%m-%dT%H:%M:%SZ', # ISO mit Z
        '%d.%m.%Y %H:%M:%S',  # Deutsche Format
        '%m/%d/%Y %H:%M:%S',  # US Format
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    
    return None

def calculate_distance_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Berechnet die Entfernung zwischen zwei GPS-Punkten in Metern (Haversine-Formel)"""
    from math import radians, cos, sin, asin, sqrt
    
    # Konvertiere zu Radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Differenzen
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine Formel
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Erdradius in Metern
    r = 6371000
    
    return c * r

def group_images_by_location(images_data: list, max_distance_meters: float = 100) -> Dict[str, list]:
    """Gruppiert Bilder nach Standort"""
    location_groups = {}
    
    for img_data in images_data:
        gps = img_data.get('metadata', {}).get('gps')
        if not gps:
            continue
            
        lat, lon = gps['lat'], gps['lon']
        
        # Suche nach bestehender Gruppe
        assigned_group = None
        for group_id, group_images in location_groups.items():
            group_lat, group_lon = group_images[0]['metadata']['gps']['lat'], group_images[0]['metadata']['gps']['lon']
            distance = calculate_distance_between_points(lat, lon, group_lat, group_lon)
            
            if distance <= max_distance_meters:
                assigned_group = group_id
                break
        
        if assigned_group:
            location_groups[assigned_group].append(img_data)
        else:
            # Neue Gruppe erstellen
            group_id = f"location_{len(location_groups) + 1}"
            location_groups[group_id] = [img_data]
    
    return location_groups

def group_images_by_time(images_data: list, max_time_diff_hours: float = 24) -> Dict[str, list]:
    """Gruppiert Bilder nach Aufnahmezeit"""
    time_groups = {}
    
    for img_data in images_data:
        datetime_str = img_data.get('metadata', {}).get('datetime')
        if not datetime_str:
            continue
            
        dt = parse_datetime_string(datetime_str)
        if not dt:
            continue
        
        # Suche nach bestehender Gruppe
        assigned_group = None
        for group_id, group_images in time_groups.items():
            group_dt_str = group_images[0]['metadata']['datetime']
            group_dt = parse_datetime_string(group_dt_str)
            
            if group_dt:
                time_diff = abs((dt - group_dt).total_seconds() / 3600)  # Stunden
                if time_diff <= max_time_diff_hours:
                    assigned_group = group_id
                    break
        
        if assigned_group:
            time_groups[assigned_group].append(img_data)
        else:
            # Neue Gruppe erstellen
            group_id = f"time_{len(time_groups) + 1}"
            time_groups[group_id] = [img_data]
    
    return time_groups
