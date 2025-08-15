import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any

from app.utils import (
    group_images_by_location, 
    group_images_by_time, 
    parse_datetime_string,
    calculate_distance_between_points
)

st.title("📊 Erweiterte Metadaten-Analyse")
st.caption("Analysieren Sie Ihre Foto-Metadaten mit Statistiken und Visualisierungen")

# Sidebar für Einstellungen
with st.sidebar:
    st.header("⚙️ Analyse-Einstellungen")
    
    # Gruppierungseinstellungen
    st.subheader("Gruppierung")
    location_threshold = st.slider("Standort-Gruppierung (Meter)", 50, 500, 100, 50)
    time_threshold = st.slider("Zeit-Gruppierung (Stunden)", 1, 72, 24, 1)
    
    # Filter
    st.subheader("Filter")
    min_quality = st.slider("Min. Bildqualität", 0.0, 1.0, 0.0, 0.1)
    min_faces = st.slider("Min. Anzahl Gesichter", 0, 10, 0, 1)
    
    # Datei-Upload
    st.subheader("Daten")
    results_file = st.file_uploader("JSON-Ergebnisse hochladen", type=["json"])

def load_and_filter_data(json_data):
    """Lädt und filtert die JSON-Daten"""
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Filter anwenden
    filtered_data = []
    for item in data:
        # Qualitätsfilter
        if min_quality > 0:
            persons = item.get('persons', [])
            max_quality = max([p.get('quality_score', 0) for p in persons]) if persons else 0
            if max_quality < min_quality:
                continue
        
        # Gesichter-Filter
        if min_faces > 0:
            if len(item.get('persons', [])) < min_faces:
                continue
        
        filtered_data.append(item)
    
    return filtered_data

def create_metadata_summary(data):
    """Erstellt eine Zusammenfassung der Metadaten"""
    summary = {
        'total_images': len(data),
        'images_with_gps': 0,
        'images_with_faces': 0,
        'images_with_camera_info': 0,
        'total_faces': 0,
        'unique_people': set(),
        'camera_models': {},
        'date_range': {'earliest': None, 'latest': None},
        'locations': [],
        'quality_scores': []
    }
    
    for item in data:
        metadata = item.get('metadata', {})
        persons = item.get('persons', [])
        
        # GPS
        if metadata.get('gps'):
            summary['images_with_gps'] += 1
            summary['locations'].append(metadata['gps'])
        
        # Gesichter
        if persons:
            summary['images_with_faces'] += 1
            summary['total_faces'] += len(persons)
            
            for person in persons:
                if person.get('name'):
                    summary['unique_people'].add(person['name'])
                if person.get('quality_score'):
                    summary['quality_scores'].append(person['quality_score'])
        
        # Kamera-Info
        if metadata.get('camera_model'):
            summary['images_with_camera_info'] += 1
            model = metadata['camera_model']
            summary['camera_models'][model] = summary['camera_models'].get(model, 0) + 1
        
        # Datum
        if metadata.get('datetime'):
            dt = parse_datetime_string(metadata['datetime'])
            if dt:
                if summary['date_range']['earliest'] is None or dt < summary['date_range']['earliest']:
                    summary['date_range']['earliest'] = dt
                if summary['date_range']['latest'] is None or dt > summary['date_range']['latest']:
                    summary['date_range']['latest'] = dt
    
    summary['unique_people'] = len(summary['unique_people'])
    return summary

def display_summary_cards(summary):
    """Zeigt Zusammenfassungskarten an"""
    st.subheader("📈 Übersicht")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gesamtbilder", summary['total_images'])
    
    with col2:
        st.metric("Mit GPS", summary['images_with_gps'])
    
    with col3:
        st.metric("Mit Gesichtern", summary['images_with_faces'])
    
    with col4:
        st.metric("Erkannte Personen", summary['unique_people'])
    
    # Qualitätsstatistiken
    if summary['quality_scores']:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ø Gesichtsqualität", f"{np.mean(summary['quality_scores']):.2f}")
        with col2:
            st.metric("Beste Qualität", f"{max(summary['quality_scores']):.2f}")
        with col3:
            st.metric("Schlechteste Qualität", f"{min(summary['quality_scores']):.2f}")

def create_face_analysis_charts(data):
    """Erstellt Charts für Gesichtsanalyse"""
    st.subheader("👥 Gesichtsanalyse")
    
    # Daten sammeln
    face_data = []
    for item in data:
        for person in item.get('persons', []):
            face_data.append({
                'age': person.get('age'),
                'gender': person.get('gender'),
                'quality': person.get('quality_score'),
                'emotion': person.get('emotion'),
                'eye_status': person.get('eye_status'),
                'mouth_status': person.get('mouth_status'),
                'image': item.get('image', 'Unknown')
            })
    
    if not face_data:
        st.info("Keine Gesichtsdaten verfügbar")
        return
    
    df = pd.DataFrame(face_data)
    
    # Altersverteilung
    if 'age' in df.columns and df['age'].notna().any():
        col1, col2 = st.columns(2)
        
        with col1:
            age_data = df[df['age'].notna()]
            fig_age = px.histogram(age_data, x='age', nbins=20, 
                                 title="Altersverteilung",
                                 labels={'age': 'Alter', 'count': 'Anzahl'})
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            gender_age = df[df['age'].notna() & df['gender'].notna()]
            if not gender_age.empty:
                fig_gender_age = px.box(gender_age, x='gender', y='age',
                                      title="Alter nach Geschlecht",
                                      labels={'gender': 'Geschlecht', 'age': 'Alter'})
                st.plotly_chart(fig_gender_age, use_container_width=True)
    
    # Qualitätsverteilung
    if 'quality' in df.columns and df['quality'].notna().any():
        col1, col2 = st.columns(2)
        
        with col1:
            fig_quality = px.histogram(df[df['quality'].notna()], x='quality', nbins=20,
                                     title="Gesichtsqualitätsverteilung",
                                     labels={'quality': 'Qualität', 'count': 'Anzahl'})
            st.plotly_chart(fig_quality, use_container_width=True)
        
        with col2:
            # Qualität nach Geschlecht
            quality_gender = df[df['quality'].notna() & df['gender'].notna()]
            if not quality_gender.empty:
                fig_quality_gender = px.box(quality_gender, x='gender', y='quality',
                                          title="Qualität nach Geschlecht",
                                          labels={'gender': 'Geschlecht', 'quality': 'Qualität'})
                st.plotly_chart(fig_quality_gender, use_container_width=True)

def create_camera_analysis_charts(data):
    """Erstellt Charts für Kamera-Analyse"""
    st.subheader("📷 Kamera-Analyse")
    
    # Kamera-Daten sammeln
    camera_data = []
    for item in data:
        metadata = item.get('metadata', {})
        if metadata.get('camera_model'):
            camera_data.append({
                'model': metadata['camera_model'],
                'make': metadata.get('camera_make', 'Unknown'),
                'lens': metadata.get('lens'),
                'focal_length': metadata.get('focal_length'),
                'f_number': metadata.get('f_number'),
                'iso': metadata.get('iso'),
                'exposure_time': metadata.get('exposure_time'),
                'datetime': metadata.get('datetime')
            })
    
    if not camera_data:
        st.info("Keine Kameradaten verfügbar")
        return
    
    df = pd.DataFrame(camera_data)
    
    # Kamera-Modelle
    if 'model' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            model_counts = df['model'].value_counts().head(10)
            fig_models = px.bar(x=model_counts.values, y=model_counts.index, orientation='h',
                              title="Top 10 Kamera-Modelle",
                              labels={'x': 'Anzahl', 'y': 'Modell'})
            st.plotly_chart(fig_models, use_container_width=True)
        
        with col2:
            # Brennweiten-Verteilung
            if 'focal_length' in df.columns and df['focal_length'].notna().any():
                focal_data = df[df['focal_length'].notna()]
                fig_focal = px.histogram(focal_data, x='focal_length', nbins=20,
                                       title="Brennweiten-Verteilung",
                                       labels={'focal_length': 'Brennweite (mm)', 'count': 'Anzahl'})
                st.plotly_chart(fig_focal, use_container_width=True)
    
    # Aufnahme-Einstellungen
    if any(col in df.columns for col in ['f_number', 'iso', 'exposure_time']):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'f_number' in df.columns and df['f_number'].notna().any():
                f_data = df[df['f_number'].notna()]
                fig_f = px.histogram(f_data, x='f_number', nbins=15,
                                   title="Blenden-Verteilung",
                                   labels={'f_number': 'f-Number', 'count': 'Anzahl'})
                st.plotly_chart(fig_f, use_container_width=True)
        
        with col2:
            if 'iso' in df.columns and df['iso'].notna().any():
                iso_data = df[df['iso'].notna()]
                fig_iso = px.histogram(iso_data, x='iso', nbins=15,
                                     title="ISO-Verteilung",
                                     labels={'iso': 'ISO', 'count': 'Anzahl'})
                st.plotly_chart(fig_iso, use_container_width=True)

def create_temporal_analysis_charts(data):
    """Erstellt Charts für zeitliche Analyse"""
    st.subheader("🕒 Zeitliche Analyse")
    
    # Zeitdaten sammeln
    time_data = []
    for item in data:
        metadata = item.get('metadata', {})
        if metadata.get('datetime'):
            dt = parse_datetime_string(metadata['datetime'])
            if dt:
                time_data.append({
                    'datetime': dt,
                    'date': dt.date(),
                    'hour': dt.hour,
                    'weekday': dt.strftime('%A'),
                    'month': dt.strftime('%B'),
                    'year': dt.year,
                    'image': item.get('image', 'Unknown'),
                    'face_count': len(item.get('persons', []))
                })
    
    if not time_data:
        st.info("Keine Zeitdaten verfügbar")
        return
    
    df = pd.DataFrame(time_data)
    
    # Aufnahmen pro Tag
    col1, col2 = st.columns(2)
    
    with col1:
        daily_counts = df['date'].value_counts().sort_index()
        fig_daily = px.line(x=daily_counts.index, y=daily_counts.values,
                           title="Aufnahmen pro Tag",
                           labels={'x': 'Datum', 'y': 'Anzahl Aufnahmen'})
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Aufnahmen pro Stunde
        hourly_counts = df['hour'].value_counts().sort_index()
        fig_hourly = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                           title="Aufnahmen pro Stunde",
                           labels={'x': 'Stunde', 'y': 'Anzahl'})
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Wochentage und Monate
    col1, col2 = st.columns(2)
    
    with col1:
        weekday_counts = df['weekday'].value_counts()
        fig_weekday = px.pie(values=weekday_counts.values, names=weekday_counts.index,
                            title="Aufnahmen nach Wochentag")
        st.plotly_chart(fig_weekday, use_container_width=True)
    
    with col2:
        month_counts = df['month'].value_counts()
        fig_month = px.pie(values=month_counts.values, names=month_counts.index,
                          title="Aufnahmen nach Monat")
        st.plotly_chart(fig_month, use_container_width=True)

def create_location_analysis_charts(data):
    """Erstellt Charts für Standort-Analyse"""
    st.subheader("📍 Standort-Analyse")
    
    # GPS-Daten sammeln
    location_data = []
    for item in data:
        metadata = item.get('metadata', {})
        if metadata.get('gps'):
            location_data.append({
                'lat': metadata['gps']['lat'],
                'lon': metadata['gps']['lon'],
                'altitude': metadata['gps'].get('altitude'),
                'image': item.get('image', 'Unknown'),
                'datetime': metadata.get('datetime'),
                'face_count': len(item.get('persons', []))
            })
    
    if not location_data:
        st.info("Keine Standortdaten verfügbar")
        return
    
    df = pd.DataFrame(location_data)
    
    # Karte
    if len(df) > 0:
        fig_map = px.scatter_mapbox(df, lat='lat', lon='lon', 
                                  hover_data=['image', 'face_count'],
                                  title="Aufnahmeorte",
                                  mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Höhenverteilung
    if 'altitude' in df.columns and df['altitude'].notna().any():
        col1, col2 = st.columns(2)
        
        with col1:
            alt_data = df[df['altitude'].notna()]
            fig_alt = px.histogram(alt_data, x='altitude', nbins=20,
                                 title="Höhenverteilung",
                                 labels={'altitude': 'Höhe (m)', 'count': 'Anzahl'})
            st.plotly_chart(fig_alt, use_container_width=True)
        
        with col2:
            # Höhe vs. Anzahl Gesichter
            alt_face = alt_data[alt_data['face_count'] > 0]
            if not alt_face.empty:
                fig_alt_face = px.scatter(alt_face, x='altitude', y='face_count',
                                        title="Höhe vs. Anzahl Gesichter",
                                        labels={'altitude': 'Höhe (m)', 'face_count': 'Anzahl Gesichter'})
                st.plotly_chart(fig_alt_face, use_container_width=True)

def display_grouping_analysis(data):
    """Zeigt Gruppierungsanalyse an"""
    st.subheader("📂 Gruppierungsanalyse")
    
    # Standort-Gruppierung
    location_groups = group_images_by_location(data, location_threshold)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Standort-Gruppen:** {len(location_groups)}")
        for group_id, group_images in location_groups.items():
            with st.expander(f"📍 {group_id} ({len(group_images)} Bilder)", expanded=False):
                for img in group_images[:5]:  # Zeige nur erste 5
                    st.write(f"- {img.get('image', 'Unknown')}")
                if len(group_images) > 5:
                    st.write(f"... und {len(group_images) - 5} weitere")
    
    # Zeit-Gruppierung
    time_groups = group_images_by_time(data, time_threshold)
    
    with col2:
        st.write(f"**Zeit-Gruppen:** {len(time_groups)}")
        for group_id, group_images in time_groups.items():
            with st.expander(f"🕒 {group_id} ({len(group_images)} Bilder)", expanded=False):
                for img in group_images[:5]:  # Zeige nur erste 5
                    st.write(f"- {img.get('image', 'Unknown')}")
                if len(group_images) > 5:
                    st.write(f"... und {len(group_images) - 5} weitere")

# Hauptausführung
if results_file is not None:
    try:
        # JSON laden
        json_content = results_file.read()
        data = json.loads(json_content)
        
        # Daten filtern
        filtered_data = load_and_filter_data(data)
        
        if not filtered_data:
            st.warning("Keine Daten nach den angewendeten Filtern gefunden.")
        else:
            # Zusammenfassung
            summary = create_metadata_summary(filtered_data)
            display_summary_cards(summary)
            
            # Tabs für verschiedene Analysen
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Gesichter", "📷 Kamera", "🕒 Zeit", "📍 Standort", "📂 Gruppierung"])
            
            with tab1:
                create_face_analysis_charts(filtered_data)
            
            with tab2:
                create_camera_analysis_charts(filtered_data)
            
            with tab3:
                create_temporal_analysis_charts(filtered_data)
            
            with tab4:
                create_location_analysis_charts(filtered_data)
            
            with tab5:
                display_grouping_analysis(filtered_data)
            
            # Download der analysierten Daten
            st.subheader("💾 Export")
            analysis_results = {
                'summary': summary,
                'location_groups': group_images_by_location(filtered_data, location_threshold),
                'time_groups': group_images_by_time(filtered_data, time_threshold),
                'filtered_data': filtered_data
            }
            
            st.download_button(
                "⬇️ Download Analyse-Ergebnisse",
                data=json.dumps(analysis_results, ensure_ascii=False, indent=2, default=str),
                file_name="analysis_results.json",
                mime="application/json"
            )
    
    except Exception as e:
        st.error(f"Fehler beim Laden der JSON-Datei: {e}")

else:
    st.info("📁 Laden Sie eine JSON-Datei mit Analyseergebnissen hoch, um zu starten.")
    
    # Beispiel-Daten anzeigen
    with st.expander("ℹ️ Über diese Analyse", expanded=False):
        st.markdown("""
        **Diese Analyse-Seite bietet:**
        
        📊 **Übersicht:**
        - Gesamtstatistiken Ihrer Fotos
        - Qualitätsbewertungen
        - Personen-Erkennung
        
        👥 **Gesichtsanalyse:**
        - Alters- und Geschlechtsverteilung
        - Qualitätsverteilung
        - Emotionsanalyse
        
        📷 **Kamera-Analyse:**
        - Verwendung verschiedener Kameras
        - Brennweiten-Verteilung
        - Aufnahme-Einstellungen
        
        🕒 **Zeitliche Analyse:**
        - Aufnahmen pro Tag/Stunde
        - Wochentags- und Monatsverteilung
        - Zeitliche Trends
        
        📍 **Standort-Analyse:**
        - Interaktive Karte
        - Höhenverteilung
        - Geografische Muster
        
        📂 **Gruppierung:**
        - Automatische Gruppierung nach Standort
        - Zeitliche Gruppierung
        - Ähnlichkeitsanalyse
        """)
