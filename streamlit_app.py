
import streamlit as st
st.set_page_config(page_title="Photo Metadata Suite", page_icon="📷", layout="wide")
st.title("Zeitkalkül Metadata Recognizer")
st.markdown("""
Wähle links eine Seite:

- **Enroll**: Embeddings für Personen-Erkennung erstellen
- **Annotate**: Fotos mit erweiterten Metadaten analysieren  
- **Analyze**: Erweiterte Statistiken und Visualisierungen
- **Train**: KI-Training mit Metadaten für bessere Genauigkeit

### Neue Features:

**Erweiterte Metadaten-Extraktion:**
- Vollständige EXIF-Daten (Kamera, Einstellungen, Datum)
- GPS-Koordinaten mit Höhenangabe
- Detaillierte Standort-Informationen

**Verbesserte Gesichtserkennung:**
- Qualitätsbewertung für jedes Gesicht
- Emotions-Erkennung (happy, neutral, unknown)
- Augen- und Mundstatus-Erkennung
- Pose-Schätzung

**Erweiterte Analyse:**
- Interaktive Charts und Statistiken
- Automatische Bildgruppierung nach Standort/Zeit
- Qualitätsfilter und -bewertung
- Export-Funktionen

### Optimierungen für bessere Metadaten-Erkennung:

1. **Qualitätsfilter**: Filtert Bilder nach Gesichtsqualität und -größe
2. **Erweiterte EXIF-Parsing**: Unterstützt mehr Metadaten-Formate
3. **Intelligente Gruppierung**: Gruppiert ähnliche Bilder automatisch
4. **Visualisierungen**: Zeigt Trends und Muster in Ihren Fotos
5. **Export-Funktionen**: Speichert alle Analysen für weitere Verarbeitung
""")
