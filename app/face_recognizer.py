
from __future__ import annotations
import pickle, os, glob
from typing import Dict, List, Optional
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from .utils import cosine_similarity

class FaceEngine:
    def __init__(self, det_size=(640,640)):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1, det_size=det_size)

    def analyze(self, img_bgr):
        faces = self.app.get(img_bgr)
        results = []
        for f in faces:
            box = f.bbox.astype(int).tolist()
            prob = float(getattr(f, "det_score", 1.0))
            gender = getattr(f, "gender", None)
            gender_str = "male" if gender == 0 else ("female" if gender == 1 else None)
            age = int(getattr(f, "age", -1)) if getattr(f, "age", None) is not None else None
            emb = f.embedding.astype(np.float32)
            
            # Erweiterte Attribute
            face_attributes = self._extract_face_attributes(f, img_bgr, box)
            
            results.append({
                "bbox": box,
                "prob": prob,
                "embedding": emb,
                "age": age if age and age >= 0 else None,
                "gender": gender_str,
                **face_attributes
            })
        return results
    
    def _extract_face_attributes(self, face, img_bgr, bbox):
        """Extrahiert erweiterte Gesichtsattribute"""
        attributes = {}
        
        # Pose-Schätzung (falls verfügbar)
        if hasattr(face, 'pose'):
            attributes['pose'] = {
                'yaw': float(getattr(face.pose, 'yaw', 0)),
                'pitch': float(getattr(face.pose, 'pitch', 0)),
                'roll': float(getattr(face.pose, 'roll', 0))
            }
        
        # Landmarks für Qualitätsbewertung
        if hasattr(face, 'kps') and face.kps is not None:
            landmarks = face.kps.astype(np.int32)
            attributes['landmarks'] = landmarks.tolist()
            
            # Qualitätsbewertung basierend auf Landmarks
            quality_score = self._assess_face_quality(img_bgr, bbox, landmarks)
            attributes['quality_score'] = quality_score
        
        # Emotion-Schätzung (einfache Implementierung)
        emotion = self._estimate_emotion(img_bgr, bbox)
        if emotion:
            attributes['emotion'] = emotion
        
        # Augen-Status
        eye_status = self._detect_eye_status(img_bgr, bbox)
        if eye_status:
            attributes['eye_status'] = eye_status
        
        # Mund-Status
        mouth_status = self._detect_mouth_status(img_bgr, bbox)
        if mouth_status:
            attributes['mouth_status'] = mouth_status
        
        return attributes
    
    def _assess_face_quality(self, img_bgr, bbox, landmarks):
        """Bewertet die Qualität des Gesichts"""
        try:
            x1, y1, x2, y2 = bbox
            face_roi = img_bgr[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return 0.0
            
            # Größe des Gesichts
            face_area = (x2 - x1) * (y2 - y1)
            img_area = img_bgr.shape[0] * img_bgr.shape[1]
            size_score = min(face_area / img_area * 100, 1.0)
            
            # Schärfe (Laplacian Variance)
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500, 1.0)  # Normalisiert
            
            # Helligkeit
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Kontrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 50, 1.0)
            
            # Gesamtqualität
            quality = (size_score * 0.3 + sharpness_score * 0.3 + 
                      brightness_score * 0.2 + contrast_score * 0.2)
            
            return float(max(0.0, min(1.0, quality)))
            
        except Exception:
            return 0.5
    
    def _estimate_emotion(self, img_bgr, bbox):
        """Einfache Emotionsschätzung basierend auf Gesichtsgeometrie"""
        try:
            x1, y1, x2, y2 = bbox
            face_roi = img_bgr[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            # Graustufen-Konvertierung
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Haar Cascade für Augen
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
            
            # Haar Cascade für Mund
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            mouths = mouth_cascade.detectMultiScale(gray, 1.1, 3)
            
            # Einfache Emotionslogik
            if len(eyes) >= 2 and len(mouths) > 0:
                return "happy"
            elif len(eyes) >= 2:
                return "neutral"
            else:
                return "unknown"
                
        except Exception:
            return None
    
    def _detect_eye_status(self, img_bgr, bbox):
        """Erkennt Augen-Status (offen/geschlossen)"""
        try:
            x1, y1, x2, y2 = bbox
            face_roi = img_bgr[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
            
            if len(eyes) >= 2:
                return "open"
            elif len(eyes) == 1:
                return "partially_open"
            else:
                return "closed"
                
        except Exception:
            return None
    
    def _detect_mouth_status(self, img_bgr, bbox):
        """Erkennt Mund-Status (offen/geschlossen)"""
        try:
            x1, y1, x2, y2 = bbox
            face_roi = img_bgr[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            mouths = mouth_cascade.detectMultiScale(gray, 1.1, 3)
            
            if len(mouths) > 0:
                return "open"
            else:
                return "closed"
                
        except Exception:
            return None

class GalleryDB:
    def __init__(self):
        self.people: Dict[str, List[np.ndarray]] = {}
        self.face_metadata: Dict[str, List[Dict]] = {}  # Erweiterte Metadaten

    def add(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        self.people.setdefault(name, []).append(embedding.astype(np.float32))
        if metadata:
            self.face_metadata.setdefault(name, []).append(metadata)

    def save(self, path: str):
        data = {
            'people': self.people,
            'metadata': self.face_metadata
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'GalleryDB':
        db = GalleryDB()
        with open(path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                db.people = data.get('people', {})
                db.face_metadata = data.get('metadata', {})
            else:
                # Rückwärtskompatibilität
                db.people = data
        return db

    def match(self, embedding: np.ndarray, threshold: float = 0.55):
        best_name, best_sim = None, -1.0
        for name, embs in self.people.items():
            sims = [cosine_similarity(embedding, e) for e in embs]
            if sims:
                sim = float(np.mean(sims))
                if sim > best_sim:
                    best_sim, best_name = sim, name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim
    
    def get_person_metadata(self, name: str) -> List[Dict]:
        """Gibt Metadaten für eine Person zurück"""
        return self.face_metadata.get(name, [])

def build_gallery_from_folder(gallery_dir: str, det_size=(640,640)) -> 'GalleryDB':
    engine = FaceEngine(det_size=det_size)
    db = GalleryDB()
    exts = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
    for person in sorted(os.listdir(gallery_dir)):
        person_dir = os.path.join(gallery_dir, person)
        if not os.path.isdir(person_dir):
            continue
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(person_dir, f"*{ext}")))
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = engine.analyze(img)
            if not faces:
                continue
            faces.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
            
            # Erweiterte Metadaten speichern
            face_data = faces[0]
            metadata = {
                'age': face_data.get('age'),
                'gender': face_data.get('gender'),
                'quality_score': face_data.get('quality_score'),
                'emotion': face_data.get('emotion'),
                'eye_status': face_data.get('eye_status'),
                'mouth_status': face_data.get('mouth_status'),
                'source_image': p
            }
            
            db.add(person, face_data["embedding"], metadata)
    return db
