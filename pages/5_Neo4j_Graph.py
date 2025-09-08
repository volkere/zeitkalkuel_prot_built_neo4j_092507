
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

# Hinweis: Früherer Annotate-spezifischer Code am Dateiende wurde entfernt
