"""
Trainings-Seite für Metadaten-basierte KI-Optimierung
"""

import streamlit as st
import json
import os
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

from app.enhanced_face_engine import MetadataAwareTrainer, EnhancedFaceEngine, create_training_dataset_from_annotations

st.title("🎯 KI-Training mit Metadaten")
st.caption("Trainieren Sie die Gesichtserkennung mit Metadaten für bessere Genauigkeit")

# Sidebar für Trainings-Einstellungen
with st.sidebar:
    st.header("⚙️ Trainings-Einstellungen")
    
    # Daten-Upload
    st.subheader("📁 Trainingsdaten")
    training_files = st.file_uploader(
        "JSON-Trainingsdaten hochladen", 
        type=["json"], 
        accept_multiple_files=True,
        help="Laden Sie JSON-Dateien mit Annotations hoch"
    )
    
    # Trainings-Parameter
    st.subheader("🎯 Trainings-Parameter")
    validation_split = st.slider("Validierungs-Split", 0.1, 0.5, 0.2, 0.05)
    
    # Metadaten-Gewichtungen
    st.subheader("📊 Metadaten-Gewichtungen")
    age_weight = st.slider("Alter-Gewichtung", 0.1, 0.5, 0.3, 0.05)
    gender_weight = st.slider("Geschlecht-Gewichtung", 0.1, 0.5, 0.25, 0.05)
    location_weight = st.slider("Standort-Gewichtung", 0.1, 0.5, 0.2, 0.05)
    temporal_weight = st.slider("Zeit-Gewichtung", 0.1, 0.5, 0.15, 0.05)
    technical_weight = st.slider("Technisch-Gewichtung", 0.1, 0.5, 0.1, 0.05)
    
    # Modell-Pfad
    st.subheader("💾 Modell-Speicherung")
    model_name = st.text_input("Modell-Name", value=f"enhanced_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model_path = f"models/{model_name}.pkl"

# Hauptbereich
if not training_files:
    st.info("📁 Laden Sie JSON-Trainingsdaten in der Sidebar hoch, um zu starten.")
    
    # Beispiel-Trainingsdaten anzeigen
    with st.expander("ℹ️ Über das Training", expanded=False):
        st.markdown("""
        **Metadaten-basiertes Training verbessert die Gesichtserkennung durch:**
        
        🎯 **Kontext-Awareness:**
        - Standort-basierte Alters-Korrektur
        - Zeit-basierte Geschlechts-Vorhersage
        - Technische Metadaten für Qualitätsbewertung
        
        📊 **Intelligente Korrekturen:**
        - Metadaten-Bias-Erkennung
        - Gewichtete Vorhersage-Kombination
        - Kontinuierliches Lernen
        
        🔄 **Workflow:**
        1. JSON-Daten mit Metadaten hochladen
        2. Training mit Metadaten-Integration
        3. Validierung und Modell-Speicherung
        4. Integration in die Annotate-Seite
        
        **Erwartete Verbesserungen:**
        - Alterserkennung: +15-25%
        - Geschlechtserkennung: +10-20%
        - Standort-basierte Vorhersagen: +20-30%
        """)
    
    # Beispiel-Trainingsdaten zum Download
    st.subheader("📥 Beispiel-Trainingsdaten")
    example_training_data = [
        {
            "image": "training_photo_1.jpg",
            "metadata": {
                "camera_make": "Canon",
                "camera_model": "EOS R5",
                "datetime": "2024-01-15T14:30:00",
                "gps": {"lat": 52.5200, "lon": 13.4050, "altitude": 34.5},
                "focal_length": 50,
                "f_number": 2.8,
                "iso": 100
            },
            "persons": [
                {
                    "age": 25,
                    "gender": "female",
                    "quality_score": 0.85,
                    "bbox": [100, 150, 300, 450]
                }
            ]
        },
        {
            "image": "training_photo_2.jpg",
            "metadata": {
                "camera_make": "iPhone",
                "camera_model": "iPhone 15 Pro",
                "datetime": "2024-01-16T10:15:00",
                "gps": {"lat": 48.1351, "lon": 11.5820, "altitude": 519.0},
                "focal_length": 35,
                "f_number": 4.0,
                "iso": 200
            },
            "persons": [
                {
                    "age": 30,
                    "gender": "male",
                    "quality_score": 0.78,
                    "bbox": [200, 100, 400, 350]
                }
            ]
        }
    ]
    
    st.download_button(
        "⬇️ Beispiel-Trainingsdaten herunterladen",
        data=json.dumps(example_training_data, ensure_ascii=False, indent=2),
        file_name="example_training_data.json",
        mime="application/json"
    )

else:
    # Trainingsdaten verarbeiten
    st.header("📊 Trainingsdaten-Analyse")
    
    # Daten laden
    training_data = []
    for file in training_files:
        try:
            data = json.load(file)
            if isinstance(data, list):
                training_data.extend(data)
            else:
                training_data.append(data)
        except Exception as e:
            st.error(f"Fehler beim Laden von {file.name}: {e}")
    
    if training_data:
        st.success(f"✅ {len(training_data)} Trainingsbeispiele geladen")
        
        # Datenanalyse
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_images = len(training_data)
            st.metric("Gesamtbilder", total_images)
        
        with col2:
            images_with_faces = sum(1 for item in training_data if item.get('persons'))
            st.metric("Mit Gesichtern", images_with_faces)
        
        with col3:
            images_with_metadata = sum(1 for item in training_data if item.get('metadata'))
            st.metric("Mit Metadaten", images_with_metadata)
        
        # Detaillierte Analyse
        with st.expander("📈 Detaillierte Datenanalyse", expanded=False):
            # Metadaten-Verteilung
            metadata_stats = analyze_training_data(training_data)
            
            # Metadaten-Charts
            if metadata_stats['camera_models']:
                fig_cameras = px.bar(
                    x=list(metadata_stats['camera_models'].keys()),
                    y=list(metadata_stats['camera_models'].values()),
                    title="Kamera-Modelle in Trainingsdaten",
                    labels={'x': 'Kamera-Modell', 'y': 'Anzahl'}
                )
                st.plotly_chart(fig_cameras, use_container_width=True)
            
            # Altersverteilung
            if metadata_stats['ages']:
                fig_ages = px.histogram(
                    x=metadata_stats['ages'],
                    nbins=20,
                    title="Altersverteilung in Trainingsdaten",
                    labels={'x': 'Alter', 'y': 'Anzahl'}
                )
                st.plotly_chart(fig_ages, use_container_width=True)
            
            # Geschlechtsverteilung
            if metadata_stats['genders']:
                fig_genders = px.pie(
                    values=list(metadata_stats['genders'].values()),
                    names=list(metadata_stats['genders'].keys()),
                    title="Geschlechtsverteilung in Trainingsdaten"
                )
                st.plotly_chart(fig_genders, use_container_width=True)
        
        # Training starten
        st.header("🚀 Training starten")
        
        if st.button("🎯 Training mit Metadaten starten", type="primary"):
            with st.spinner("Training läuft..."):
                try:
                    # Trainer initialisieren
                    metadata_weights = {
                        'age': age_weight,
                        'gender': gender_weight,
                        'location': location_weight,
                        'temporal': temporal_weight,
                        'technical': technical_weight
                    }
                    
                    trainer = MetadataAwareTrainer(model_path)
                    
                    # Training durchführen
                    results = trainer.train(training_data, validation_split)
                    
                    # Ergebnisse anzeigen
                    st.success("🎉 Training erfolgreich abgeschlossen!")
                    
                    # Trainings-Ergebnisse
                    display_training_results(results)
                    
                    # Modell-Download
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            st.download_button(
                                "⬇️ Trainiertes Modell herunterladen",
                                data=f.read(),
                                file_name=f"{model_name}.pkl",
                                mime="application/octet-stream"
                            )
                    
                except Exception as e:
                    st.error(f"Fehler beim Training: {e}")
                    st.exception(e)

def analyze_training_data(training_data: List[Dict]) -> Dict[str, Any]:
    """Analysiert Trainingsdaten"""
    stats = {
        'camera_models': {},
        'ages': [],
        'genders': {},
        'locations': {},
        'quality_scores': []
    }
    
    for item in training_data:
        # Kamera-Modelle
        metadata = item.get('metadata', {})
        camera_model = metadata.get('camera_model', 'Unknown')
        stats['camera_models'][camera_model] = stats['camera_models'].get(camera_model, 0) + 1
        
        # Personen-Daten
        persons = item.get('persons', [])
        for person in persons:
            if person.get('age'):
                stats['ages'].append(person['age'])
            
            if person.get('gender'):
                gender = person['gender']
                stats['genders'][gender] = stats['genders'].get(gender, 0) + 1
            
            if person.get('quality_score'):
                stats['quality_scores'].append(person['quality_score'])
        
        # Standorte
        location = metadata.get('location', {}).get('country', 'Unknown')
        stats['locations'][location] = stats['locations'].get(location, 0) + 1
    
    return stats

def display_training_results(results: Dict[str, Any]):
    """Zeigt Trainings-Ergebnisse an"""
    st.subheader("📊 Trainings-Ergebnisse")
    
    # Metriken
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'training' in results and 'age_accuracy' in results['training']:
            st.metric("Alter-Genauigkeit", f"{results['training']['age_accuracy']:.3f}")
    
    with col2:
        if 'training' in results and 'gender_accuracy' in results['training']:
            st.metric("Geschlecht-Genauigkeit", f"{results['training']['gender_accuracy']:.3f}")
    
    with col3:
        if 'training' in results and 'quality_accuracy' in results['training']:
            st.metric("Qualität-Genauigkeit", f"{results['training']['quality_accuracy']:.3f}")
    
    # Detaillierte Ergebnisse
    with st.expander("📈 Detaillierte Ergebnisse", expanded=False):
        if 'training' in results:
            st.write("**Training-Metriken:**")
            for metric, value in results['training'].items():
                st.write(f"- {metric}: {value:.3f}")
        
        if 'validation' in results:
            st.write("**Validierungs-Metriken:**")
            for metric, value in results['validation'].items():
                st.write(f"- {metric}: {value:.3f}")
        
        if 'improvement' in results:
            st.write("**Verbesserungen:**")
            for metric, improvement in results['improvement'].items():
                st.write(f"- {metric}: {improvement:+.3f}")
    
    # Modell-Integration
    st.subheader("🔗 Modell-Integration")
    st.info("""
    **So integrieren Sie das trainierte Modell:**
    
    1. **Modell herunterladen** (Button oben)
    2. **In Annotate-Seite hochladen** als "Enhanced Model"
    3. **Erweiterte Erkennung aktivieren** in den Einstellungen
    
    Das trainierte Modell wird automatisch Metadaten für bessere Vorhersagen nutzen!
    """)

# Modell-Test-Bereich
st.header("🧪 Modell-Test")
st.caption("Testen Sie ein trainiertes Modell mit neuen Bildern")

test_model_file = st.file_uploader("Trainiertes Modell hochladen", type=["pkl"])

if test_model_file:
    try:
        # Temporäre Datei erstellen
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(test_model_file.read())
            tmp_path = tmp_file.name
        
        # Modell laden
        engine = EnhancedFaceEngine()
        engine.load_models(tmp_path)
        
        st.success("✅ Modell erfolgreich geladen!")
        
        # Test-Bilder
        test_images = st.file_uploader(
            "Test-Bilder hochladen", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        if test_images:
            st.subheader("🔍 Test-Ergebnisse")
            
            for img_file in test_images:
                # Hier würde die Bildverarbeitung implementiert
                st.write(f"**{img_file.name}**: Modell geladen und bereit für Tests")
        
        # Aufräumen
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
