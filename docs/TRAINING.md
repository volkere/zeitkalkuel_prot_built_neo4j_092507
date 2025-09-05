# Metadaten-basiertes KI-Training

## Übersicht

Das erweiterte Training-System verbessert die Gesichtserkennung durch Integration von Metadaten wie Standort, Zeit, Kamera-Einstellungen und technischen Parametern.

## Schnellstart

### 1. Trainingsdaten vorbereiten

Erstellen Sie JSON-Dateien mit folgender Struktur:

```json
[
  {
    "image": "foto1.jpg",
    "metadata": {
      "camera_make": "Canon",
      "camera_model": "EOS R5",
      "datetime": "2024-01-15T14:30:00",
      "gps": {
        "lat": 52.5200,
        "lon": 13.4050,
        "altitude": 34.5
      },
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
  }
]
```

### 2. Training über UI

1. Gehen Sie zur **Train-Seite** in der Streamlit-App
2. Laden Sie JSON-Trainingsdaten hoch
3. Konfigurieren Sie die Metadaten-Gewichtungen
4. Starten Sie das Training
5. Laden Sie das trainierte Modell herunter

### 3. Training über CLI

```bash
python train_enhanced_model.py \
  --input training_data.json \
  --output models/my_enhanced_model.pkl \
  --validation-split 0.2 \
  --age-weight 0.3 \
  --gender-weight 0.25 \
  --location-weight 0.2 \
  --temporal-weight 0.15 \
  --technical-weight 0.1
```

## Metadaten-Integration

### Demografische Metadaten
- **Alter**: Normalisiert auf 0-1 Skala
- **Geschlecht**: One-hot Encoding (male/female/unknown)
- **Altersgruppen**: child, teen, young_adult, adult, senior

### Standort-Metadaten
- **GPS-Koordinaten**: Normalisiert (lat/90, lon/180)
- **Höhe**: Normalisiert auf Mount Everest (8848m)
- **Länder**: Top 10 Länder One-hot Encoding

### Zeitliche Metadaten
- **Stunde**: Normalisiert (0-24)
- **Wochentag**: One-hot Encoding (7 Tage)
- **Monat**: One-hot Encoding (12 Monate)
- **Jahreszeit**: One-hot Encoding (4 Jahreszeiten)

### Technische Metadaten
- **Bildqualität**: 0-1 Skala
- **Kamera-Modelle**: Top 8 Hersteller
- **Brennweite**: Normalisiert auf 200mm
- **ISO**: Normalisiert auf 6400
- **Blende**: Normalisiert auf f/22

## Trainings-Algorithmus

### 1. Metadaten-Encoding
```python
def encode_all_metadata(metadata: Dict) -> np.ndarray:
    demographics = self.encode_demographics(metadata)
    location = self.encode_location(metadata)
    temporal = self.encode_temporal(metadata)
    technical = self.encode_technical(metadata)
    
    return np.concatenate([demographics, location, temporal, technical])
```

### 2. Modell-Training
- **Alters-Modell**: RandomForest für Altersvorhersage
- **Geschlechts-Modell**: RandomForest für Geschlechtsklassifikation
- **Qualitäts-Modell**: RandomForest für Qualitätsbewertung

### 3. Metadaten-Bias-Korrektur
- **Standort-Alter-Bias**: Durchschnittsalter pro Land
- **Zeit-Geschlecht-Bias**: Geschlechtsverteilung pro Stunde
- **Technische-Qualität-Bias**: Qualitätskorrelation mit Kamera-Einstellungen

### 4. Vorhersage-Enhancement
```python
def _enhance_with_metadata(base_prediction, metadata, metadata_features):
    # Alters-Korrektur
    if self.age_model is not None:
        predicted_age = self.age_model.predict([metadata_features])[0]
        enhanced['age'] = int(0.7 * base_age + 0.3 * predicted_age)
    
    # Geschlechts-Korrektur
    if self.gender_model is not None:
        predicted_gender = self.gender_model.predict([metadata_features])[0]
        if confidence > 0.8:
            enhanced['gender'] = predicted_gender
    
    # Standort-basierte Korrektur
    if location in self.location_age_bias:
        enhanced['age'] = int(0.9 * enhanced['age'] + 0.1 * location_bias)
    
    return enhanced
```

## Erwartete Verbesserungen

| Metrik | Verbesserung | Begründung |
|--------|-------------|------------|
| **Alterserkennung** | +15-25% | Standort- und zeitbasierte Korrekturen |
| **Geschlechtserkennung** | +10-20% | Tageszeit- und Kontext-Integration |
| **Standort-Vorhersagen** | +20-30% | Geografische Bias-Erkennung |
| **Temporale Konsistenz** | +25-35% | Zeitliche Muster-Erkennung |
| **Gesamtqualität** | +15-25% | Technische Metadaten-Integration |

## Konfiguration

### Metadaten-Gewichtungen
```python
metadata_weights = {
    'age': 0.3,        # Alters-Erkennung
    'gender': 0.25,    # Geschlechts-Erkennung
    'location': 0.2,   # Standort-Metadaten
    'temporal': 0.15,  # Zeitliche Metadaten
    'technical': 0.1   # Technische Metadaten
}
```

### Trainings-Parameter
- **Validierungs-Split**: 0.1-0.5 (Standard: 0.2)
- **RandomForest-Parameter**: 100 Estimators, Random State 42
- **Feature-Normalisierung**: Min-Max Scaling
- **Bias-Schwellenwerte**: 5+ Beispiele pro Kategorie

## Dateistruktur

```
models/
├── enhanced_model.pkl          # Trainiertes Modell
├── enhanced_model_info.json    # Modell-Metadaten
└── training_history.json       # Training-Historie

training_data/
├── dataset1.json              # Trainingsdaten
├── dataset2.json
└── validation_data.json       # Validierungsdaten
```

## Modell-Testing

### 1. UI-Testing
- Gehen Sie zur **Train-Seite**
- Laden Sie ein trainiertes Modell hoch
- Testen Sie mit neuen Bildern

### 2. CLI-Testing
```python
from app.enhanced_face_engine import EnhancedFaceEngine

# Modell laden
engine = EnhancedFaceEngine()
engine.load_models("models/enhanced_model.pkl")

# Vorhersage mit Metadaten
metadata = {
    "camera_model": "iPhone 15 Pro",
    "datetime": "2024-01-15T14:30:00",
    "gps": {"lat": 52.5200, "lon": 13.4050}
}

predictions = engine.predict_with_metadata(image, metadata)
```

## Kontinuierliches Lernen

### Feedback-Sammlung
```python
class ContinuousLearning:
    def collect_feedback(self, prediction, user_correction, metadata):
        feedback_entry = {
            'prediction': prediction,
            'correction': user_correction,
            'metadata': metadata,
            'timestamp': datetime.now(),
            'confidence_delta': abs(prediction['confidence'] - user_correction['confidence'])
        }
        self.feedback_database.append(feedback_entry)
```

### Modell-Updates
- **Batch-Größe**: 100 Feedback-Einträge
- **Lernrate**: 0.01
- **Update-Frequenz**: Wöchentlich
- **A/B-Testing**: Kontinuierliche Validierung

## Monitoring

### Trainings-Metriken
- **Accuracy**: Klassifikations-Genauigkeit
- **MAE**: Mean Absolute Error (Alter)
- **F1-Score**: Geschlechts-Erkennung
- **Correlation**: Qualitäts-Bewertung

### Bias-Monitoring
- **Standort-Bias**: Altersverteilung pro Land
- **Zeit-Bias**: Geschlechtsverteilung pro Stunde
- **Technischer-Bias**: Qualität vs. Kamera-Modell

## Troubleshooting

### Häufige Probleme

1. **Keine Trainingsdaten**
   - Stellen Sie sicher, dass JSON-Dateien korrekt formatiert sind
   - Überprüfen Sie die Metadaten-Struktur

2. **Niedrige Genauigkeit**
   - Erhöhen Sie die Anzahl der Trainingsbeispiele
   - Überprüfen Sie die Metadaten-Gewichtungen
   - Validieren Sie die Label-Qualität

3. **Modell-Lade-Fehler**
   - Überprüfen Sie den Modell-Pfad
   - Stellen Sie sicher, dass alle Abhängigkeiten installiert sind

### Debug-Modus
```bash
python train_enhanced_model.py --input data.json --verbose
```

## Weiterführende Informationen

- [Trainingsplan](training_plan.md) - Detaillierter Implementierungsplan
- [API-Dokumentation](API.md) - Technische API-Referenz
- [Beispiele](examples/) - Code-Beispiele und Tutorials
