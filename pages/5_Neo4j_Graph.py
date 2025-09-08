
import io
import json
import os
import pandas as pd
import streamlit as st

from typing import List, Dict, Any
from streamlit.components.v1 import html as st_html
from app.graph_visualizer import GraphVisualizer

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

    tab_info, tab_data, tab_import, tab_query, tab_vis, tab_explore, tab_admin = st.tabs([
        "Datenbank-Info", "Daten-View", "Import", "Abfragen", "Visualisierung", "Explore", "Verwaltung"
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

    with tab_data:
        st.subheader("Daten-View")
        
        # Data type selector
        data_type = st.selectbox("Datentyp anzeigen:", [
            "Alle Personen", "Alle Bilder", "Alle Standorte", "Alle Gesichter", 
            "Alle Kameras", "Alle Aufnahmen", "Alle Adressen", "Alle Tänze"
        ])
        
        # Limit selector
        limit = st.slider("Anzahl Einträge", 10, 1000, 100, 10)
        
        if data_type == "Alle Personen":
            query = "MATCH (p:Person) RETURN p.name as Name, p ORDER BY p.name LIMIT $limit"
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame([{"Name": r.get("Name", "Unbekannt")} for r in result])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Personen gefunden")
        
        elif data_type == "Alle Bilder":
            query = """
            MATCH (i:Image) 
            OPTIONAL MATCH (i)-[:AT_LOCATION]->(l:Location)
            OPTIONAL MATCH (i)-[:HAS_CAMERA]->(c:Camera)
            RETURN i.name as Bildname, i.width as Breite, i.height as Höhe, 
                   l.lat as Breitengrad, l.lon as Längengrad,
                   c.make as Kamera_Hersteller, c.model as Kamera_Modell
            ORDER BY i.name LIMIT $limit
            """
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Bilder gefunden")
        
        elif data_type == "Alle Standorte":
            query = """
            MATCH (l:Location) 
            OPTIONAL MATCH (l)-[:RESOLVED_AS]->(a:Address)
            RETURN l.lat as Breitengrad, l.lon as Längengrad, l.altitude as Höhe,
                   a.full_address as Adresse, a.city as Stadt, a.country as Land
            ORDER BY l.lat, l.lon LIMIT $limit
            """
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Standorte gefunden")
        
        elif data_type == "Alle Gesichter":
            query = """
            MATCH (f:Face)
            OPTIONAL MATCH (f)-[:IDENTIFIED_AS]->(p:Person)
            OPTIONAL MATCH (f)<-[:CONTAINS]-(i:Image)
            RETURN f.emotion as Emotion, f.quality_score as Qualität, 
                   p.name as Person, i.name as Bild
            ORDER BY f.quality_score DESC LIMIT $limit
            """
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Gesichter gefunden")
        
        elif data_type == "Alle Kameras":
            query = """
            MATCH (c:Camera)
            RETURN c.make as Hersteller, c.model as Modell, c.lens as Objektiv
            ORDER BY c.make, c.model LIMIT $limit
            """
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Kameras gefunden")
        
        elif data_type == "Alle Aufnahmen":
            query = """
            MATCH (cap:Capture)
            RETURN cap.datetime as Datum_Zeit, cap.hour as Stunde, 
                   cap.weekday as Wochentag, cap.part_of_day as Tageszeit
            ORDER BY cap.datetime DESC LIMIT $limit
            """
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Aufnahmen gefunden")
        
        elif data_type == "Alle Adressen":
            query = """
            MATCH (a:Address)
            RETURN a.full_address as Vollständige_Adresse, a.city as Stadt, 
                   a.state as Bundesland, a.country as Land, a.postcode as PLZ
            ORDER BY a.city, a.country LIMIT $limit
            """
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Adressen gefunden")
        
        elif data_type == "Alle Tänze":
            query = """
            MATCH (d:Dance)
            OPTIONAL MATCH (f:Face)-[r:PERFORMS]->(d)
            RETURN d.label as Tanz_Art, count(r) as Anzahl_Ausführungen
            ORDER BY count(r) DESC LIMIT $limit
            """
            result = neo.execute_cypher(query, {"limit": limit})
            if result and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Keine Tänze gefunden")
        
        # Summary statistics
        with st.expander("Zusammenfassung", expanded=False):
            st.write("**Datenbank-Statistiken:**")
            info = neo.get_database_info()
            if "error" not in info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gesamt Knoten", sum(info.get("node_counts", {}).values()))
                with col2:
                    st.metric("Gesamt Beziehungen", sum(info.get("relationship_counts", {}).values()))
                with col3:
                    st.metric("Knotentypen", len(info.get("node_counts", {})))

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
        
        # Example queries
        example_queries = {
            "Alle Knoten zählen": "MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt ORDER BY cnt DESC",
            "Personen mit Bildern": "MATCH (p:Person)<-[:IDENTIFIED_AS]-(f:Face)<-[:CONTAINS]-(i:Image) RETURN p.name, count(i) as image_count",
            "Bilder an Standorten": "MATCH (i:Image)-[:AT_LOCATION]->(l:Location) RETURN l.lat, l.lon, count(i) as photo_count",
            "Gesichter mit Emotionen": "MATCH (f:Face) WHERE f.emotion IS NOT NULL RETURN f.emotion, count(f) as count",
            "Komplexe Abfrage": "MATCH (p:Person)<-[:IDENTIFIED_AS]-(f:Face)<-[:CONTAINS]-(i:Image)-[:AT_LOCATION]->(l:Location) WHERE f.emotion = 'happy' RETURN p.name, l.lat, l.lon, count(i) as happy_photos"
        }
        
        selected_example = st.selectbox("Beispiel-Query wählen:", ["Eigene Query"] + list(example_queries.keys()))
        
        if selected_example == "Eigene Query":
            q = st.text_area("Cypher", value="MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt ORDER BY cnt DESC")
        else:
            q = st.text_area("Cypher", value=example_queries[selected_example])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Query ausführen"):
                res = neo.execute_cypher(q)
                try:
                    st.dataframe(pd.DataFrame(res))
                except Exception:
                    st.json(res)
        
        with col2:
            if st.button("Query visualisieren"):
                gv = GraphVisualizer()
                query_viz = gv.visualize_cypher_query(q)
                st_html(query_viz, height=600)
        
        # Show parsed query structure
        with st.expander("Query-Struktur analysieren", expanded=False):
            gv = GraphVisualizer()
            parsed = gv.parse_cypher_query(q)
            if "error" in parsed:
                st.error(parsed["error"])
            else:
                st.write("**Gefundene Knoten:**")
                for node in parsed["nodes"]:
                    st.write(f"- {node['id']}: {', '.join(node['labels'])}")
                
                st.write("**Gefundene Beziehungen:**")
                for rel in parsed["relationships"]:
                    st.write(f"- {rel['type']}")
                
                if parsed["return_variables"]:
                    st.write(f"**RETURN:** {', '.join(parsed['return_variables'])}")
                
                if parsed["where_conditions"]:
                    st.write(f"**WHERE:** {parsed['where_conditions'][0]}")

    with tab_vis:
        st.subheader("Graph-Visualisierung")
        limit = st.slider("Limit", 100, 10000, 2000, 100)
        viz_type = st.radio("Visualisierungstyp", ["Interaktiv (pyvis)", "Statisch (matplotlib)"], index=0)
        
        if viz_type == "Interaktiv (pyvis)":
            show_buttons = st.checkbox("Steuerung anzeigen (Physik/Nodes/Edges)", True)
            physics = st.checkbox("Physik aktiv", True)
            scale_cent = st.checkbox("Größe nach Zentralität", True)
            min_deg = st.slider("Minimaler Knotengrad (Filter)", 0, 5, 0)
            show_edge_labels = st.checkbox("Kanten-Labels anzeigen", True)

        data = neo.get_graph_data(limit=limit)
        if "error" in data:
            st.error(data["error"])
        else:
            # Debug information
            st.write(f"**Geladene Daten:** {len(data.get('nodes', []))} Knoten, {len(data.get('relationships', []))} Beziehungen")
            
            if not data.get("nodes"):
                st.warning("Keine Knoten gefunden. Bitte importieren Sie zuerst Daten über den Import-Tab.")
            else:
                # Show sample data
                with st.expander("Beispieldaten anzeigen", expanded=False):
                    st.write("**Erste 3 Knoten:**")
                    for i, node in enumerate(data.get("nodes", [])[:3]):
                        st.write(f"{i+1}. ID: {node['id']}, Labels: {node['labels']}, Properties: {list(node['properties'].keys())}")
                    
                    st.write("**Erste 3 Beziehungen:**")
                    for i, rel in enumerate(data.get("relationships", [])[:3]):
                        st.write(f"{i+1}. {rel['source']} -[{rel['type']}]-> {rel['target']}")
                
                gv = GraphVisualizer(height="700px")
                
                if viz_type == "Interaktiv (pyvis)":
                    html_body = gv.create_interactive_network(
                        data,
                        show_buttons=show_buttons,
                        scale_by_centrality=scale_cent,
                        physics=physics,
                        min_degree=min_deg,
                        show_edge_labels=show_edge_labels,
                    )
                    
                    if html_body:
                        st_html(html_body, height=740)
                    else:
                        st.error("Fehler beim Generieren der interaktiven Visualisierung")
                else:
                    # Static visualization
                    static_html = gv.create_static_network(data, "Neo4j Graph")
                    st_html(static_html, height=600)

    with tab_explore:
        st.subheader("Explore: Interaktive Nachbarschaft")
        # State
        if "explore_nodes" not in st.session_state:
            st.session_state["explore_nodes"] = set()
        if "explore_graph" not in st.session_state:
            st.session_state["explore_graph"] = {"nodes": [], "relationships": []}

        col_a, col_b = st.columns([2, 1])
        with col_a:
            term = st.text_input("Suche (Name/Label/Eigenschaft)")
            if st.button("Suchen") and term.strip():
                results = neo.search_nodes(term.strip(), limit=50)
                if results and not isinstance(results[0], dict) and results[0].get("error"):
                    st.error(results[0]["error"])  # unlikely path
                else:
                    st.write("Treffer:")
                    for r in results:
                        st.write(f"ID {r['id']} | Labels: {', '.join(r['labels'])} | Props: {list(r['properties'].keys())[:5]}")
        with col_b:
            node_id_str = st.text_input("Node-ID fokussieren")
            depth = st.slider("Tiefe", 1, 3, 1)
            if st.button("Fokussieren/Erweitern"):
                try:
                    nid = int(node_id_str)
                    sub = neo.get_neighborhood(nid, depth=depth, limit=2000)
                    if "error" in sub:
                        st.error(sub["error"])
                    else:
                        # Merge into explore graph
                        g = st.session_state["explore_graph"]
                        existing_ids = {n["id"] for n in g["nodes"]}
                        for n in sub.get("nodes", []):
                            if n["id"] not in existing_ids:
                                g["nodes"].append(n)
                        g["relationships"].extend(sub.get("relationships", []))
                        st.session_state["explore_nodes"].add(nid)
                        st.success(f"Subgraph (Tiefe {depth}) hinzugefügt.")
                except ValueError:
                    st.warning("Bitte eine gültige numerische Node-ID angeben.")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Zurücksetzen"):
                st.session_state["explore_graph"] = {"nodes": [], "relationships": []}
                st.session_state["explore_nodes"] = set()
        with c2:
            physics_e = st.checkbox("Physik aktiv (Explore)", True)
        with c3:
            show_labels_e = st.checkbox("Kanten-Labels (Explore)", False)

        eg = st.session_state["explore_graph"]
        if eg["nodes"]:
            gv = GraphVisualizer(height="750px")
            html_body = gv.create_interactive_network(
                eg,
                show_buttons=True,
                physics=physics_e,
                scale_by_centrality=True,
                min_degree=0,
                show_edge_labels=show_labels_e,
                highlight_nodes=st.session_state.get("explore_nodes")
            )
            st_html(html_body, height=780)
        else:
            st.info("Suche einen Knoten und erweitere seine Nachbarschaft, um das Explore-Netzwerk aufzubauen.")

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
