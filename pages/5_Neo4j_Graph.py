
import io
import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        
        # View type selector
        view_type = st.radio("Ansicht:", ["Tabellen", "Karten", "Diagramme", "Interaktive Graphen"], horizontal=True)
        
        # Data type selector
        data_type = st.selectbox("Datentyp anzeigen:", [
            "Alle Personen", "Alle Bilder", "Alle Standorte", "Alle Gesichter", 
            "Alle Kameras", "Alle Aufnahmen", "Alle Adressen", "Alle Tänze"
        ])
        
        # Limit selector
        limit = st.slider("Anzahl Einträge", 10, 1000, 100, 10)
        
        # Test visualization
        if st.button("Test-Visualisierung anzeigen"):
            st.write("**Test-Diagramm:**")
            test_data = {"Name": ["Alice", "Bob", "Charlie"], "Count": [1, 2, 3]}
            fig = px.bar(x=test_data["Name"], y=test_data["Count"], title="Test-Diagramm")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Test-Netzwerk:**")
            test_network = {
                "nodes": [
                    {"id": "test1", "labels": ["Person"], "properties": {"name": "Test Person 1"}},
                    {"id": "test2", "labels": ["Person"], "properties": {"name": "Test Person 2"}}
                ],
                "relationships": [
                    {"source": "test1", "target": "test2", "type": "KNOWS"}
                ]
            }
            gv = GraphVisualizer(height="400px")
            html_body = gv.create_interactive_network(test_network, "Test-Netzwerk")
            if html_body:
                st_html(html_body, height=420)
            else:
                st.error("Test-Netzwerk konnte nicht generiert werden")
        
        if data_type == "Alle Personen":
            query = "MATCH (p:Person) RETURN p.name as Name, p ORDER BY p.name LIMIT $limit"
            result = neo.execute_cypher(query, {"limit": limit})
            
            # Debug information
            st.write(f"**Debug:** Query: {query}")
            st.write(f"**Debug:** Result type: {type(result)}, Length: {len(result) if result else 0}")
            if result:
                st.write(f"**Debug:** First result: {result[0] if result else 'None'}")
            
            if result and len(result) > 0 and not (isinstance(result[0], dict) and "error" in result[0]):
                df = pd.DataFrame([{"Name": r.get("Name", "Unbekannt")} for r in result])
                st.write(f"**Debug:** DataFrame created with {len(df)} rows")
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Diagramme":
                    # Person count chart
                    if len(df) > 0:
                        fig = px.bar(df, x="Name", y=[1]*len(df), title="Personen in der Datenbank")
                        fig.update_layout(showlegend=False, yaxis_title="Anzahl")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Keine Daten für Diagramm verfügbar")
                elif view_type == "Interaktive Graphen":
                    # Use real subgraph with relationships
                    sub = neo.get_subgraph_by_label("Person", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Personen gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Personen-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
                else:
                    st.info("Karten-Ansicht nicht verfügbar für Personen")
            else:
                if result and len(result) > 0 and isinstance(result[0], dict) and "error" in result[0]:
                    st.error(f"Query-Fehler: {result[0]['error']}")
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
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Karten":
                    # Filter out images without coordinates
                    df_map = df.dropna(subset=['Breitengrad', 'Längengrad'])
                    if not df_map.empty:
                        fig = px.scatter_mapbox(
                            df_map, 
                            lat="Breitengrad", 
                            lon="Längengrad",
                            hover_name="Bildname",
                            hover_data=["Breite", "Höhe", "Kamera_Hersteller", "Kamera_Modell"],
                            color="Kamera_Hersteller",
                            size_max=15,
                            zoom=1,
                            title="Bilder auf der Karte"
                        )
                        fig.update_layout(
                            mapbox_style="open-street-map",
                            height=600,
                            margin={"r":0,"t":30,"l":0,"b":0}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Keine Bilder mit GPS-Koordinaten gefunden")
                elif view_type == "Diagramme":
                    # Resolution distribution
                    if 'Breite' in df.columns and 'Höhe' in df.columns:
                        df_res = df.dropna(subset=['Breite', 'Höhe'])
                        if not df_res.empty:
                            df_res['Auflösung'] = df_res['Breite'].astype(str) + 'x' + df_res['Höhe'].astype(str)
                            res_counts = df_res['Auflösung'].value_counts().head(10)
                            fig = px.bar(x=res_counts.index, y=res_counts.values, title="Häufigste Auflösungen")
                            fig.update_layout(xaxis_title="Auflösung", yaxis_title="Anzahl", xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Camera manufacturer distribution
                    if 'Kamera_Hersteller' in df.columns:
                        cam_counts = df['Kamera_Hersteller'].value_counts()
                        fig = px.pie(values=cam_counts.values, names=cam_counts.index, title="Kamera-Hersteller Verteilung")
                        st.plotly_chart(fig, use_container_width=True)
                elif view_type == "Interaktive Graphen":
                    sub = neo.get_subgraph_by_label("Image", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Bilder gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Bilder-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
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
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Karten":
                    # Filter out locations without coordinates
                    df_map = df.dropna(subset=['Breitengrad', 'Längengrad'])
                    if not df_map.empty:
                        fig = px.scatter_mapbox(
                            df_map, 
                            lat="Breitengrad", 
                            lon="Längengrad",
                            hover_name="Stadt",
                            hover_data=["Adresse", "Land", "Höhe"],
                            color="Land",
                            size_max=15,
                            zoom=1,
                            title="Standorte auf der Karte"
                        )
                        fig.update_layout(
                            mapbox_style="open-street-map",
                            height=600,
                            margin={"r":0,"t":30,"l":0,"b":0}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Keine Standorte mit GPS-Koordinaten gefunden")
                elif view_type == "Diagramme":
                    # Country distribution
                    if 'Land' in df.columns:
                        country_counts = df['Land'].value_counts().head(10)
                        fig = px.pie(values=country_counts.values, names=country_counts.index, title="Verteilung nach Ländern")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Altitude distribution
                    if 'Höhe' in df.columns:
                        df_alt = df.dropna(subset=['Höhe'])
                        if not df_alt.empty:
                            fig = px.histogram(df_alt, x="Höhe", title="Höhenverteilung", nbins=20)
                            st.plotly_chart(fig, use_container_width=True)
                elif view_type == "Interaktive Graphen":
                    sub = neo.get_subgraph_by_label("Location", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Standorte gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Standorte-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
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
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Diagramme":
                    # Emotion distribution
                    if 'Emotion' in df.columns:
                        emotion_counts = df['Emotion'].value_counts()
                        fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, title="Emotionsverteilung")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Quality distribution
                    if 'Qualität' in df.columns:
                        df_qual = df.dropna(subset=['Qualität'])
                        if not df_qual.empty:
                            fig = px.histogram(df_qual, x="Qualität", title="Qualitätsverteilung", nbins=20)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Person vs Emotion heatmap
                    if 'Person' in df.columns and 'Emotion' in df.columns:
                        df_heat = df.dropna(subset=['Person', 'Emotion'])
                        if not df_heat.empty:
                            heatmap_data = df_heat.groupby(['Person', 'Emotion']).size().unstack(fill_value=0)
                            fig = px.imshow(heatmap_data, title="Personen vs Emotionen", aspect="auto")
                            st.plotly_chart(fig, use_container_width=True)
                elif view_type == "Interaktive Graphen":
                    sub = neo.get_subgraph_by_label("Face", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Gesichter gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Gesichter-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
                else:
                    st.info("Karten-Ansicht nicht verfügbar für Gesichter")
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
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Diagramme":
                    # Camera manufacturer distribution
                    if 'Hersteller' in df.columns:
                        make_counts = df['Hersteller'].value_counts()
                        fig = px.bar(x=make_counts.index, y=make_counts.values, title="Kamera-Hersteller Verteilung")
                        fig.update_layout(xaxis_title="Hersteller", yaxis_title="Anzahl")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Lens distribution
                    if 'Objektiv' in df.columns:
                        lens_counts = df['Objektiv'].value_counts().head(10)
                        fig = px.pie(values=lens_counts.values, names=lens_counts.index, title="Objektiv-Verteilung")
                        st.plotly_chart(fig, use_container_width=True)
                elif view_type == "Interaktive Graphen":
                    sub = neo.get_subgraph_by_label("Camera", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Kameras gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Kamera-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
                else:
                    st.info("Karten-Ansicht nicht verfügbar für Kameras")
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
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Diagramme":
                    # Hour distribution
                    if 'Stunde' in df.columns:
                        hour_counts = df['Stunde'].value_counts().sort_index()
                        fig = px.bar(x=hour_counts.index, y=hour_counts.values, title="Aufnahmen nach Stunde")
                        fig.update_layout(xaxis_title="Stunde", yaxis_title="Anzahl")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Weekday distribution
                    if 'Wochentag' in df.columns:
                        weekday_counts = df['Wochentag'].value_counts()
                        fig = px.pie(values=weekday_counts.values, names=weekday_counts.index, title="Aufnahmen nach Wochentag")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Time of day distribution
                    if 'Tageszeit' in df.columns:
                        time_counts = df['Tageszeit'].value_counts()
                        fig = px.bar(x=time_counts.index, y=time_counts.values, title="Aufnahmen nach Tageszeit")
                        fig.update_layout(xaxis_title="Tageszeit", yaxis_title="Anzahl")
                        st.plotly_chart(fig, use_container_width=True)
                elif view_type == "Interaktive Graphen":
                    sub = neo.get_subgraph_by_label("Capture", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Aufnahmen gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Aufnahmen-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
                else:
                    st.info("Karten-Ansicht nicht verfügbar für Aufnahmen")
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
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Diagramme":
                    # Country distribution
                    if 'Land' in df.columns:
                        country_counts = df['Land'].value_counts()
                        fig = px.pie(values=country_counts.values, names=country_counts.index, title="Adressen nach Ländern")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # City distribution
                    if 'Stadt' in df.columns:
                        city_counts = df['Stadt'].value_counts().head(15)
                        fig = px.bar(x=city_counts.index, y=city_counts.values, title="Top 15 Städte")
                        fig.update_layout(xaxis_title="Stadt", yaxis_title="Anzahl", xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                elif view_type == "Interaktive Graphen":
                    sub = neo.get_subgraph_by_label("Address", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Adressen gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Adressen-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
                else:
                    st.info("Karten-Ansicht nicht verfügbar für Adressen")
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
                
                if view_type == "Tabellen":
                    st.dataframe(df, use_container_width=True)
                elif view_type == "Diagramme":
                    # Dance type distribution
                    if 'Tanz_Art' in df.columns and 'Anzahl_Ausführungen' in df.columns:
                        fig = px.bar(df, x="Tanz_Art", y="Anzahl_Ausführungen", title="Tanz-Ausführungen")
                        fig.update_layout(xaxis_title="Tanz-Art", yaxis_title="Anzahl Ausführungen", xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Pie chart for dance types
                        fig = px.pie(df, values="Anzahl_Ausführungen", names="Tanz_Art", title="Verteilung der Tanz-Arten")
                        st.plotly_chart(fig, use_container_width=True)
                elif view_type == "Interaktive Graphen":
                    sub = neo.get_subgraph_by_label("Dance", node_limit=limit)
                    if "error" in sub:
                        st.error(sub["error"])
                    elif not sub.get("nodes"):
                        st.info("Kein Subgraph für Tänze gefunden")
                    else:
                        gv = GraphVisualizer(height="650px")
                        html_body = gv.create_interactive_network(sub, "Tanz-Subgraph", show_edge_labels=True)
                        st_html(html_body, height=680)
                else:
                    st.info("Karten-Ansicht nicht verfügbar für Tänze")
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
