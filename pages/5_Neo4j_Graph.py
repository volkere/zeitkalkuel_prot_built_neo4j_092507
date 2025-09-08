
import io
import json
import os
import pandas as pd
import streamlit as st

from typing import List, Dict, Any

from app.graph_persistence import Neo4jPersistence


st.title("Neo4j Graph-Datenbank")
st.caption("Verbinden, importieren, abfragen und visualisieren")

# Verbindung (Sidebar)
with st.sidebar:
    st.subheader("Verbindung")
    with st.form("neo4j_connection_form", clear_on_submit=False):
        uri = st.text_input("Bolt URI", value=st.session_state.get("neo4j_uri", "bolt://localhost:7687"))
        user = st.text_input("User", value=st.session_state.get("neo4j_user", "neo4j"))
        password = st.text_input("Passwort", type="password", value=st.session_state.get("neo4j_pwd", ""))
        submitted = st.form_submit_button("Verbinden")
        if submitted:
            try:
                neo = Neo4jPersistence(uri, user, password)
                if neo.test_connection():
                    st.session_state["neo4j_conn"] = neo
                    st.session_state["neo4j_uri"] = uri
                    st.session_state["neo4j_user"] = user
                    st.session_state["neo4j_pwd"] = password
                    st.success("Verbunden")
                else:
                    st.error("Verbindung fehlgeschlagen")
            except Exception as e:
                st.error(f"Fehler: {e}")
    if st.button("Trennen"):
        if st.session_state.get("neo4j_conn"):
            try:
                st.session_state["neo4j_conn"].close()
            except Exception:
                pass
        st.session_state.pop("neo4j_conn", None)
        st.success("Getrennt")


connected = st.session_state.get("neo4j_conn") is not None
st.info("Status: Verbunden" if connected else "Status: Nicht verbunden")

if connected:
    neo: Neo4jPersistence = st.session_state["neo4j_conn"]

    tab_info, tab_import, tab_query, tab_vis, tab_admin = st.tabs([
        "Datenbank-Info", "Import", "Abfragen", "Visualisierung", "Verwaltung"
    ])

    with tab_info:
        st.subheader("Übersicht")
        info = neo.get_database_info()
        if "error" in info:
            st.error(info["error"])
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.write("Knoten pro Label")
                st.dataframe(pd.DataFrame([{"label": k, "count": v} for k, v in info.get("node_counts", {}).items()]))
            with c2:
                st.write("Beziehungen pro Typ")
                st.dataframe(pd.DataFrame([{"type": k, "count": v} for k, v in info.get("relationship_counts", {}).items()]))

    with tab_import:
        st.subheader("JSON-Import")
        up = st.file_uploader("Annotations-JSON hochladen (Liste von Records)", type=["json"])
        if up is not None:
            try:
                data = json.load(up)
                if isinstance(data, dict):
                    data = data.get("records") or data.get("results") or []
                if not isinstance(data, list):
                    st.error("Unerwartetes JSON-Format. Erwartet: Liste von Objekten.")
                else:
                    neo.init_constraints()
                    neo.upsert_annotations(data)
                    st.success(f"{len(data)} Datensätze importiert")
            except Exception as e:
                st.error(f"Fehler beim Import: {e}")

    with tab_query:
        st.subheader("Cypher ausführen")
        q = st.text_area("Cypher", value="MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt ORDER BY cnt DESC")
        if st.button("Query ausführen"):
            res = neo.execute_cypher(q)
            try:
                st.dataframe(pd.DataFrame(res))
            except Exception:
                st.json(res)

    with tab_vis:
        st.subheader("Graph-Visualisierung (vereinfachte Ansicht)")
        limit = st.slider("Limit", 100, 5000, 500, 100)
        data = neo.get_graph_data(limit=limit)
        if "error" in data:
            st.error(data["error"])
        else:
            st.caption(f"Nodes: {len(data.get('nodes', []))}, Relationships: {len(data.get('relationships', []))}")
            st.json(data)

    with tab_admin:
        st.subheader("Verwaltung")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Graph als JSON exportieren"):
                import tempfile
                tmp = tempfile.mktemp(suffix=".json")
                if neo.export_to_json(tmp):
                    with open(tmp, "rb") as f:
                        st.download_button("Download JSON", data=f.read(), file_name="graph.json", mime="application/json")
                else:
                    st.error("Export fehlgeschlagen")
        with c2:
            if st.button("Datenbank leeren (ALLE Daten)"):
                if neo.clear_database():
                    st.warning("Alle Daten wurden gelöscht.")
                else:
                    st.error("Löschen fehlgeschlagen")
else:
    st.info("Bitte zuerst verbinden, um Funktionen anzuzeigen.")


@st.cache_resource(show_spinner=False)
def _load_clip_for_dance():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor, device


def _prepare_dance_text_embeddings(labels: list):
    import torch
    model, processor, device = _load_clip_for_dance()
    prompts = [f"a person performing {lbl.strip()} dance" for lbl in labels]
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, device, model, processor


def _infer_dance_for_face(pil_image, text_features, device, model, processor, labels: list):
    import torch
    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        img_feat = model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = img_feat @ text_features.T
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        top_idx = int(np.argmax(probs))
        return labels[top_idx], float(probs[top_idx])

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
            st.info(f"{metadata['datetime']}")
        
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
                st.write("Identifikation:")
                if person.get("name"):
                    st.success(f"{person['name']} (Ähnlichkeit: {person.get('similarity', 0):.2f})")
                else:
                    st.warning("Unbekannte Person")
                
                # Demografie
                st.write("Demografie:")
                if person.get("age"):
                    st.write(f"Alter: {person['age']} Jahre")
                if person.get("gender"):
                    st.write(f"Geschlecht: {person['gender']}")
                
                # Qualität
                if person.get("quality_score"):
                    quality = person['quality_score']
                    st.write("Qualität:")
                    if quality > 0.7:
                        st.success(f"Hohe Qualität ({quality:.2f})")
                    elif quality > 0.4:
                        st.warning(f"Mittlere Qualität ({quality:.2f})")
                    else:
                        st.error(f"Niedrige Qualität ({quality:.2f})")
            
            with col2:
                # Emotion und Status
                st.write("Gesichtsausdruck:")
                if person.get("emotion"):
                    st.write(f"{person['emotion']}")
                
                # Augen-Status
                if person.get("eye_status"):
                    st.write(f"Augen: {person['eye_status']}")
                
                # Mund-Status
                if person.get("mouth_status"):
                    st.write(f"Mund: {person['mouth_status']}")

results: List[Dict[str, Any]] = []
annotated_images = []  # (filename, image_bytes)

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
        # Score-Filter und Limit
        faces = [f for f in faces if f.get('prob', 1.0) >= min_det_score]
        if max_faces and len(faces) > max_faces:
            faces.sort(key=lambda f: f.get('prob', 0.0), reverse=True)
            faces = faces[:max_faces]
        
        # Qualitätsfilter anwenden
        filtered_faces = []
        for f in faces:
            face_size = (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])
            if (f.get('quality_score', 0.5) >= min_quality and 
                face_size >= min_face_size):
                filtered_faces.append(f)
        
        persons = []
        # Dance setup (cached per label set)
        dance_labels = [l.strip() for l in dance_labels_text.split(',') if l.strip()]
        if enable_dance and dance_labels:
            key = ("dance_text_embeds", tuple(dance_labels))
            cache = st.session_state.get("dance_cache") or {}
            if cache.get("labels_key") != key:
                text_features, device, model, processor = _prepare_dance_text_embeddings(dance_labels)
                st.session_state["dance_cache"] = {
                    "labels_key": key,
                    "text_features": text_features,
                    "device": device,
                    "model": model,
                    "processor": processor,
                }
            text_features = st.session_state["dance_cache"]["text_features"]
            device = st.session_state["dance_cache"]["device"]
            model = st.session_state["dance_cache"]["model"]
            processor = st.session_state["dance_cache"]["processor"]
        for f in filtered_faces:
            name, sim = (None, None)
            if db:
                n, s = db.match(f["embedding"], threshold=threshold)
                name, sim = (n, s)
            person_entry = {
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
            }

            # Dance inference per face (optional)
            if enable_dance and dance_labels:
                try:
                    x1,y1,x2,y2 = map(int, f["bbox"])
                    if dance_use_face_crop:
                        # expand bbox
                        h, w = img_bgr.shape[:2]
                        dx = int((x2 - x1) * 0.5)
                        dy = int((y2 - y1) * 0.8)
                        xx1 = max(0, x1 - dx)
                        yy1 = max(0, y1 - dy)
                        xx2 = min(w, x2 + dx)
                        yy2 = min(h, y2 + dy)
                    else:
                        xx1, yy1, xx2, yy2 = 0, 0, img_bgr.shape[1], img_bgr.shape[0]
                    crop = Image.fromarray(cv2.cvtColor(img_bgr[yy1:yy2, xx1:xx2], cv2.COLOR_BGR2RGB))
                    dance_label, dance_score = _infer_dance_for_face(crop, text_features, device, model, processor, dance_labels)
                    person_entry["dance"] = dance_label
                    person_entry["dance_score"] = dance_score
                except Exception:
                    pass

            persons.append(person_entry)

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

        # Bild-Analyse (optional)
        image_analysis = {}
        if do_quality:
            image_analysis['quality'] = assess_image_quality(img_bgr)
        if do_color_hist:
            # Nur normalisierte Histogramme in JSON
            hist = extract_color_histogram(img_bgr)
            image_analysis['color_hist'] = {k: v.tolist() for k, v in hist.items() if k.endswith('_normalized')}
        if do_composition:
            image_analysis['composition'] = analyze_image_composition(img_bgr)

        # Datum/Zeit anreichern
        if enrich_datetime and metadata.get('datetime'):
            dt = parse_datetime_string(metadata['datetime'])
            if dt:
                hour = dt.hour
                if 6 <= hour < 12:
                    part = 'morning'
                elif 12 <= hour < 17:
                    part = 'afternoon'
                elif 17 <= hour < 21:
                    part = 'evening'
                else:
                    part = 'night'
                metadata['time'] = {
                    'hour': hour,
                    'weekday': dt.strftime('%A'),
                    'part_of_day': part
                }

        # Map-Links
        if include_map_links and metadata.get('gps'):
            lat, lon = metadata['gps']['lat'], metadata['gps']['lon']
            metadata['gps']['map_links'] = {
                'openstreetmap': f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=16/{lat:.6f}/{lon:.6f}",
                'google_maps': f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
            }
        
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
            "image_analysis": image_analysis,
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
            # Optional: Zusätzliche Darstellung
            if draw_landmarks or draw_pose or blur_unknown:
                vis = boxed.copy()
                # Landmarks
                if draw_landmarks:
                    for f in filtered_faces:
                        kps = f.get('landmarks')
                        if kps:
                            for (x, y) in kps:
                                cv2.circle(vis, (int(x), int(y)), 1, (255, 0, 0), -1)
                # Pose
                if draw_pose:
                    for f in filtered_faces:
                        pose = f.get('pose')
                        if pose:
                            x1,y1,x2,y2 = map(int, f['bbox'])
                            txt = f"Y:{pose['yaw']:.1f} P:{pose['pitch']:.1f} R:{pose['roll']:.1f}"
                            cv2.putText(vis, txt, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                # Blur unbekannt
                if blur_unknown:
                    for f, p in zip(filtered_faces, persons):
                        if p.get('name') is None:
                            x1,y1,x2,y2 = map(int, f['bbox'])
                            roi = vis[y1:y2, x1:x2]
                            if roi.size > 0:
                                roi = cv2.GaussianBlur(roi, (31,31), 15)
                                vis[y1:y2, x1:x2] = roi
                boxed = vis
            rgb_vis = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
            st.image(rgb_vis, caption="Erkannte Gesichter", use_container_width=True)
            # Für ZIP-Export vormerken
            import io as _io
            buf = _io.BytesIO()
            Image.fromarray(rgb_vis).save(buf, format="JPEG", quality=90)
            annotated_images.append((up.name, buf.getvalue()))
        
        # Metadaten anzeigen
        display_metadata_card(metadata, "Bild-Metadaten")

        # Bild-Analyse anzeigen
        if image_analysis:
            with st.expander("Bild-Analyse", expanded=False):
                if image_analysis.get('quality'):
                    q = image_analysis['quality']
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Gesamtqualität", f"{q['overall_quality']:.2f}")
                    col2.metric("Schärfe", f"{q['sharpness']:.2f}")
                    col3.metric("Helligkeit", f"{q['brightness']:.2f}")
                    col4.metric("Kontrast", f"{q['contrast']:.2f}")
                    col5.metric("Rauschen", f"{q['noise_level']:.2f}")
                if image_analysis.get('composition'):
                    comp = image_analysis['composition']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Auflösung", f"{comp.get('dimensions',{}).get('width','?')}×{comp.get('dimensions',{}).get('height','?')}")
                    col2.metric("Seitenverhältnis", f"{comp.get('aspect_ratio',0):.2f}")
                    col3.metric("Farbtemp.", comp.get('color_temperature','-'))
        
        # Standort anzeigen
        if location_info:
            with st.expander("Standort-Informationen", expanded=False):
                if location_info.get('full_address'):
                    st.info(f"Adresse: {location_info['full_address']}")
                
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
                # Karten-Links, falls vorhanden
                if metadata.get('gps', {}).get('map_links'):
                    links = metadata['gps']['map_links']
                    st.markdown(f"[OpenStreetMap]({links['openstreetmap']}) | [Google Maps]({links['google_maps']})")
        
        # Gesichtsanalyse anzeigen
        display_face_analysis(persons)
        
        # JSON-Export (optional je Bild)
        if show_json_each:
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
        import zipfile as _zipfile
        import io as _io
        if annotated_images:
            zip_buf = _io.BytesIO()
            with _zipfile.ZipFile(zip_buf, mode="w", compression=_zipfile.ZIP_DEFLATED) as zf:
                for fname, data_bytes in annotated_images:
                    base = os.path.splitext(os.path.basename(fname))[0]
                    zf.writestr(f"annotated/{base}_annotated.jpg", data_bytes)
            st.download_button("Download annotated images (ZIP)", data=zip_buf.getvalue(), file_name="annotated_images.zip", mime="application/zip")
        st.info("Tipp: Laden Sie die JSON-Datei in der 'Analyze'-Seite hoch für Statistiken!")
else:
    st.info("Bilder in der Sidebar hochladen, um zu starten.")
    
    # Download-Button auch ohne Bilder (für Beispiel-Daten)
    st.subheader("Export-Optionen")
    st.info("Nach dem Hochladen und Verarbeiten von Bildern erscheint hier ein Download-Button für die JSON-Ergebnisse.")
    
    # Beispiel-Metadaten anzeigen
    with st.expander("Verfügbare Metadaten", expanded=False):
        st.markdown("""
        Diese App kann folgende Metadaten extrahieren:
        
        Kamera-Informationen:
        - Hersteller und Modell
        - Objektiv
        - Software
        
        Aufnahme-Einstellungen:
        - Brennweite
        - Blende (f-number)
        - ISO-Wert
        - Belichtungszeit
        - Weißabgleich
        - Belichtungsmodus
        
        Zeitstempel:
        - Aufnahmedatum und -zeit
        - GPS-Zeitstempel
        
        Standort:
        - GPS-Koordinaten
        - Höhe über Meeresspiegel
        - Vollständige Adresse (mit Internetverbindung)
        
        Gesichtsanalyse:
        - Alter und Geschlecht
        - Gesichtsqualität
        - Emotionen
        - Augen- und Mundstatus
        - Pose-Schätzung
        """)
