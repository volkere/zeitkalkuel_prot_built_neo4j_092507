
import io, json, tempfile
from typing import List, Dict, Any
import streamlit as st
import numpy as np
from PIL import Image
import cv2

from app.face_recognizer import FaceEngine, GalleryDB
from app.location import extract_exif_gps, reverse_geocode, extract_comprehensive_metadata, get_location_details

st.title("Annotate: Fotos analysieren")
st.caption("Erweiterte Gesichtserkennung, Metadaten-Extraktion und Standortanalyse")

# Benutzerführung
with st.expander("Anleitung", expanded=False):
    st.markdown("""
    **So verwenden Sie diese Seite:**
    
    1. **Bilder hochladen**: Wählen Sie Bilder in der Sidebar aus
    2. **Einstellungen anpassen**: Konfigurieren Sie die Erkennungsparameter
    3. **Verarbeitung**: Die App analysiert automatisch alle Bilder
    4. **Download**: Nach der Verarbeitung erscheint ein Download-Button am Ende
    5. **Analyse**: Laden Sie die JSON-Datei in der 'Analyze'-Seite hoch
    
    **Tipp**: Der Download-Button erscheint erst nach der Verarbeitung aller Bilder!
    """)

with st.sidebar:
    st.header("Einstellungen")
    
    # Gesichtserkennung
    st.subheader("Gesichtserkennung")
    det = st.slider("Detector size", 320, 1024, 640, 64, key="det_annot")
    threshold = st.slider("Identity threshold (cosine)", 0.3, 0.9, 0.55, 0.01)
    
    # Metadaten
    st.subheader("Metadaten")
    extract_full_metadata = st.checkbox("Vollständige EXIF-Metadaten extrahieren", value=True)
    do_reverse = st.checkbox("Reverse geocode GPS (Internet)", value=False)
    show_location_details = st.checkbox("Detaillierte Standortinfos", value=False)
    
    # Qualitätsfilter
    st.subheader("Qualitätsfilter")
    min_quality = st.slider("Min. Gesichtsqualität", 0.0, 1.0, 0.3, 0.1)
    min_face_size = st.slider("Min. Gesichtsgröße (Pixel)", 50, 200, 80, 10)
    
    # Datei-Upload
    st.subheader("Dateien")
    gallery_file = st.file_uploader("Embeddings DB (embeddings.pkl)", type=["pkl"], key="db_upload")
    files = st.file_uploader("Bilder hochladen", type=["jpg","jpeg","png","bmp","webp","tif","tiff"], accept_multiple_files=True)

if "engine_annot" not in st.session_state or st.session_state.get("det_annot_state") != det:
    st.session_state["engine_annot"] = FaceEngine(det_size=(det, det))
    st.session_state["det_annot_state"] = det

db = None
if gallery_file is not None:
    import pickle
    try:
        db = GalleryDB()
        data = pickle.load(gallery_file)
        if isinstance(data, dict):
            db.people = data.get('people', {})
            db.face_metadata = data.get('metadata', {})
        else:
            db.people = data
        st.success(f"Embeddings geladen: {len(db.people)} Personen.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Embeddings: {e}")

def draw_boxes(img_bgr, persons):
    img = img_bgr.copy()
    for p in persons:
        x1,y1,x2,y2 = map(int, p["bbox"])
        
        # Farbe basierend auf Qualität
        quality = p.get('quality_score', 0.5)
        if quality > 0.7:
            color = (0, 255, 0)  # Grün für hohe Qualität
        elif quality > 0.4:
            color = (0, 255, 255)  # Gelb für mittlere Qualität
        else:
            color = (0, 0, 255)  # Rot für niedrige Qualität
        
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        
        # Label mit erweiterten Informationen
        label_parts = []
        
        # Name und Ähnlichkeit
        if p.get("name"):
            sim = f" ({p['similarity']:.2f})" if p.get("similarity") is not None else ""
            label_parts.append(p["name"] + sim)
        
        # Demografie
        if p.get("gender"):
            label_parts.append(p["gender"])
        if p.get("age") is not None:
            label_parts.append(f"{p['age']}J")
        
        # Qualität
        if p.get("quality_score"):
            label_parts.append(f"Q:{p['quality_score']:.2f}")
        
        # Emotion
        if p.get("emotion"):
            label_parts.append(p["emotion"])
        
        # Augen/Mund Status
        status_parts = []
        if p.get("eye_status"):
            status_parts.append(f"Augen:{p['eye_status']}")
        if p.get("mouth_status"):
            status_parts.append(f"Mund:{p['mouth_status']}")
        
        if status_parts:
            label_parts.append(" ".join(status_parts))
        
        txt = " | ".join(label_parts) if label_parts else f"{p.get('prob', 1.0):.2f}"
        
        # Text-Hintergrund für bessere Lesbarkeit
        (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, max(0,y1-text_height-8)), (x1+text_width, y1), color, -1)
        cv2.putText(img, txt, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
    return img

def display_metadata_card(metadata, title="Metadaten"):
    """Zeigt Metadaten in einer schönen Karte an"""
    with st.expander(f"{title}", expanded=False):
        if not metadata:
            st.info("Keine Metadaten verfügbar")
            return
        
        # Kamera-Informationen
        if any(key in metadata for key in ['camera_make', 'camera_model', 'lens']):
            st.subheader("Kamera")
            col1, col2, col3 = st.columns(3)
            with col1:
                if metadata.get('camera_make'):
                    st.metric("Hersteller", metadata['camera_make'])
            with col2:
                if metadata.get('camera_model'):
                    st.metric("Modell", metadata['camera_model'])
            with col3:
                if metadata.get('lens'):
                    st.metric("Objektiv", metadata['lens'])
        
        # Aufnahme-Einstellungen
        if any(key in metadata for key in ['focal_length', 'f_number', 'iso', 'exposure_time']):
            st.subheader("Aufnahme-Einstellungen")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if metadata.get('focal_length'):
                    st.metric("Brennweite", f"{metadata['focal_length']}mm")
            with col2:
                if metadata.get('f_number'):
                    st.metric("Blende", f"f/{metadata['f_number']}")
            with col3:
                if metadata.get('iso'):
                    st.metric("ISO", metadata['iso'])
            with col4:
                if metadata.get('exposure_time'):
                    st.metric("Belichtung", f"1/{metadata['exposure_time']}s")
        
        # Datum und Zeit
        if metadata.get('datetime'):
            st.subheader("Aufnahmezeit")
            st.info(f"**{metadata['datetime']}**")
        
        # GPS und Standort
        if metadata.get('gps'):
            st.subheader("Standort")
            gps = metadata['gps']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Breitengrad", f"{gps['lat']:.6f}")
            with col2:
                st.metric("Längengrad", f"{gps['lon']:.6f}")
            
            if gps.get('altitude'):
                st.metric("Höhe", f"{gps['altitude']:.1f}m")
            
            if gps.get('timestamp'):
                st.info(f"GPS-Zeitstempel: {gps['timestamp']}")
        
        # Bildgröße
        if metadata.get('image_width') and metadata.get('image_height'):
            st.subheader("Bildgröße")
            st.metric("Auflösung", f"{metadata['image_width']} × {metadata['image_height']} Pixel")

def display_face_analysis(persons):
    """Zeigt detaillierte Gesichtsanalyse an"""
    if not persons:
        st.info("Keine Gesichter erkannt")
        return
    
    st.subheader("Gesichtsanalyse")
    
    for i, person in enumerate(persons):
        with st.expander(f"Person {i+1}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Basis-Informationen
                st.write("**Identifikation:**")
                if person.get("name"):
                    st.success(f"{person['name']} (Ähnlichkeit: {person.get('similarity', 0):.2f})")
                else:
                    st.warning("Unbekannte Person")
                
                # Demografie
                st.write("**Demografie:**")
                if person.get("age"):
                    st.write(f"Alter: {person['age']} Jahre")
                if person.get("gender"):
                    st.write(f"Geschlecht: {person['gender']}")
                
                # Qualität
                if person.get("quality_score"):
                    quality = person['quality_score']
                    st.write("**Qualität:**")
                    if quality > 0.7:
                        st.success(f"Hohe Qualität ({quality:.2f})")
                    elif quality > 0.4:
                        st.warning(f"Mittlere Qualität ({quality:.2f})")
                    else:
                        st.error(f"Niedrige Qualität ({quality:.2f})")
            
            with col2:
                # Emotion und Status
                st.write("**Gesichtsausdruck:**")
                if person.get("emotion"):
                    st.write(f"{person['emotion']}")
                
                # Augen-Status
                if person.get("eye_status"):
                    st.write(f"Augen: {person['eye_status']}")
                
                # Mund-Status
                if person.get("mouth_status"):
                    st.write(f"Mund: {person['mouth_status']}")

results: List[Dict[str, Any]] = []

if files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, up in enumerate(files):
        status_text.text(f"Verarbeite {up.name}...")
        
        data = up.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Gesichtserkennung
        faces = st.session_state["engine_annot"].analyze(img_bgr)
        
        # Qualitätsfilter anwenden
        filtered_faces = []
        for f in faces:
            face_size = (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])
            if (f.get('quality_score', 0.5) >= min_quality and 
                face_size >= min_face_size):
                filtered_faces.append(f)
        
        persons = []
        for f in filtered_faces:
            name, sim = (None, None)
            if db:
                n, s = db.match(f["embedding"], threshold=threshold)
                name, sim = (n, s)
            persons.append({
                "bbox": f["bbox"],
                "prob": f["prob"],
                "name": name,
                "similarity": sim,
                "age": f["age"],
                "gender": f["gender"],
                "quality_score": f.get("quality_score"),
                "emotion": f.get("emotion"),
                "eye_status": f.get("eye_status"),
                "mouth_status": f.get("mouth_status")
            })

        # Metadaten-Extraktion
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            image.save(tmp.name, format="JPEG")
            
            if extract_full_metadata:
                metadata = extract_comprehensive_metadata(tmp.name)
                gps_data = metadata.get('gps')
            else:
                metadata = {}
                gps_data = extract_exif_gps(tmp.name)
                if gps_data:
                    metadata['gps'] = gps_data
        
        # Standort-Informationen
        location_info = None
        if gps_data and do_reverse:
            if show_location_details:
                location_info = get_location_details(gps_data['lat'], gps_data['lon'])
            else:
                address = reverse_geocode(gps_data['lat'], gps_data['lon'])
                if address:
                    location_info = {'full_address': address}

        record = {
            "image": up.name,
            "metadata": metadata,
            "location": location_info,
            "persons": persons
        }
        results.append(record)

        # Anzeige
        st.header(f"{up.name}")
        
        # Bildanzeige
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            boxed = draw_boxes(img_bgr, persons)
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), caption="Erkannte Gesichter", use_container_width=True)
        
        # Metadaten anzeigen
        display_metadata_card(metadata, "Bild-Metadaten")
        
        # Standort anzeigen
        if location_info:
            with st.expander("Standort-Informationen", expanded=False):
                if location_info.get('full_address'):
                    st.info(f"**Adresse:** {location_info['full_address']}")
                
                if location_info.get('country'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if location_info.get('country'):
                            st.metric("Land", location_info['country'])
                    with col2:
                        if location_info.get('state'):
                            st.metric("Bundesland", location_info['state'])
                    with col3:
                        if location_info.get('city'):
                            st.metric("Stadt", location_info['city'])
        
        # Gesichtsanalyse anzeigen
        display_face_analysis(persons)
        
        # JSON-Export (kollabiert)
        with st.expander("JSON-Daten", expanded=False):
            st.json(record)
        
        st.divider()
        
        # Fortschritt aktualisieren
        progress_bar.progress((idx + 1) / len(files))
    
    status_text.text("Verarbeitung abgeschlossen!")
    
    # Download-Button für alle Ergebnisse
    st.success(f"{len(results)} Bilder erfolgreich verarbeitet")
    
    # Download-Button
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download results JSON",
                           data=json.dumps(results, ensure_ascii=False, indent=2),
                           file_name="results.json",
                           mime="application/json")
    with col2:
        st.info("Tipp: Laden Sie diese JSON-Datei in der 'Analyze'-Seite hoch für erweiterte Statistiken!")
else:
    st.info("Bilder in der Sidebar hochladen, um zu starten.")
    
    # Download-Button auch ohne Bilder (für Beispiel-Daten)
    st.subheader("Export-Optionen")
    st.info("Nach dem Hochladen und Verarbeiten von Bildern erscheint hier ein Download-Button für die JSON-Ergebnisse.")
    
    # Beispiel-Metadaten anzeigen
    with st.expander("Verfügbare Metadaten", expanded=False):
        st.markdown("""
        **Diese App kann folgende Metadaten extrahieren:**
        
        **Kamera-Informationen:**
        - Hersteller und Modell
        - Objektiv
        - Software
        
        **Aufnahme-Einstellungen:**
        - Brennweite
        - Blende (f-number)
        - ISO-Wert
        - Belichtungszeit
        - Weißabgleich
        - Belichtungsmodus
        
        **Zeitstempel:**
        - Aufnahmedatum und -zeit
        - GPS-Zeitstempel
        
        **Standort:**
        - GPS-Koordinaten
        - Höhe über Meeresspiegel
        - Vollständige Adresse (mit Internetverbindung)
        
        **Gesichtsanalyse:**
        - Alter und Geschlecht
        - Gesichtsqualität
        - Emotionen
        - Augen- und Mundstatus
        - Pose-Schätzung
        """)
