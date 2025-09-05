
Zeitkalkül agent_prot_092505

Erweiterte Foto-Metadaten-Analyse mit Gesichtserkennung, EXIF-Extraktion und intelligenten Analysen.  
(Enthält eine CLI und eine Streamlit Multi-Page UI (Enroll + Annotate + Analyze)).

Erweiterte Metadaten-Extraktion
- **Vollständige EXIF-Daten**: Kamera-Modell, Objektiv, Aufnahme-Einstellungen
- **GPS mit Höhenangabe**: Präzise Standortdaten mit Altitude
- **Detaillierte Standort-Info**: Vollständige Adressen und geografische Details
- **Zeitstempel-Parsing**: Unterstützt verschiedene Datumsformate

Verbesserte Gesichtserkennung
- **Qualitätsbewertung**: Automatische Bewertung der Gesichtsqualität
- **Emotions-Erkennung**: Happy, neutral, unknown
- **Status-Erkennung**: Augen (offen/geschlossen) und Mund-Status
- **Pose-Schätzung**: Yaw, Pitch, Roll-Winkel
- **Erweiterte Demografie**: Alters- und Geschlechtserkennung

Intelligente Analyse
- **Interaktive Visualisierungen**: Charts und Statistiken mit Plotly
- **Automatische Gruppierung**: Nach Standort und Zeit
- **Qualitätsfilter**: Filtert nach Gesichtsqualität und -größe
- **Export-Funktionen**: JSON-Export für weitere Verarbeitung

Features
- Face detection & embeddings (InsightFace `buffalo_l`)
- Age & gender estimation (approximate)
- Known-person matching via embeddings database (`embeddings.pkl`)
- EXIF GPS extraction with optional reverse geocoding
- **Erweiterte Metadaten-Extraktion** (Kamera, Einstellungen, Datum)
- **Qualitätsbewertung** für Gesichter und Bilder
- **Emotions- und Status-Erkennung**
- **Interaktive Analysen** mit Charts und Statistiken
- **Intelligente Bildgruppierung**
- Streamlit UI mit drag & drop, bounding boxes, JSON export
- CLI for batch processing

> Use responsibly:** Face analysis and attribute inference can be biased and regulated. Ensure you have the right to process the images and comply with local laws (see `docs/PRIVACY.md`).

---

Quickstart (UI)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run streamlit_app.py
```

UI-Seiten:
- **Enroll**: Erstellen von Embeddings für Personen-Erkennung
- **Annotate**: Erweiterte Foto-Analyse mit Metadaten
- **Analyze**: Statistiken, Charts und Gruppierungsanalyse

Quickstart (CLI)
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# (optional) build embeddings from a labeled gallery
python -m app.main enroll --gallery ./gallery --db embeddings.pkl

# annotate a folder with enhanced metadata
python -m app.main annotate --input ./photos --out output.json --recursive --reverse-geocode
```

Repo layout
```
app/                  # Python package (engine, CLI)
pages/                # Streamlit pages (Enroll, Annotate, Analyze)
streamlit_app.py      # Streamlit entry
requirements.txt      # runtime deps
pyproject.toml        # package metadata + console script
docs/                 # documentation
.github/workflows/    # CI (lint/build)
```

Install as a package (optional)
```bash
pip install -e .
# now the CLI is available as:
photo-meta annotate --input photos --out output.json --recursive
```

Optimierungen für bessere Metadaten-Erkennung

1. Qualitätsfilter
- **Gesichtsqualität**: Filtert nach Schärfe, Helligkeit, Kontrast
- **Größenfilter**: Mindestgröße für Gesichter
- **Qualitätsbewertung**: Automatische Bewertung von 0-1

2. Erweiterte EXIF-Parsing
- **Mehr Formate**: Unterstützt verschiedene EXIF-Standards
- **Vollständige Metadaten**: Kamera, Objektiv, Einstellungen
- **Fehlerbehandlung**: Robuste Parsing-Logik

3. Intelligente Gruppierung
- **Standort-Gruppierung**: Gruppiert Bilder in 100m-Radius
- **Zeit-Gruppierung**: Gruppiert nach 24h-Zeitfenster
- **Ähnlichkeitsanalyse**: Automatische Kategorisierung

4. Visualisierungen
- **Interaktive Charts**: Plotly-basierte Visualisierungen
- **Statistiken**: Alters-, Qualitäts-, Kamera-Verteilungen
- **Karten**: GPS-Standorte auf interaktiven Karten

5. Export-Funktionen
- **JSON-Export**: Vollständige Metadaten
- **Analyse-Export**: Gruppierungen und Statistiken
- **Format-Kompatibilität**: Standardisierte Ausgabe


