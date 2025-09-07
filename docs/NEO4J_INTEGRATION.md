# Neo4j Graph Database Integration

## Übersicht

Die Neo4j-Integration ermöglicht es, alle Foto-Metadaten in einer Graphdatenbank zu speichern und die Beziehungen zwischen Bildern, Personen, Standorten und anderen Metadaten graphisch zu visualisieren.

## Features

### Graph-Datenbank
- **Vollständige Metadaten-Speicherung**: Alle EXIF-Daten, Gesichtserkennung, Standorte und Analysen
- **Beziehungsmodellierung**: Intelligente Verknüpfung zwischen allen Entitäten
- **Skalierbare Architektur**: Unterstützt große Mengen von Fotos und Metadaten

### Visualisierung
- **Interaktive Netzwerk-Graphen**: Zoomen, verschieben, Knoten auswählen
- **Statische Visualisierungen**: Für Dokumentation und Präsentationen
- **Standort-Karten**: Geografische Darstellung der Foto-Standorte
- **Personen-Netzwerke**: Wer wurde zusammen fotografiert?

### Abfragen
- **Cypher-Interface**: Direkte Ausführung von Neo4j-Abfragen
- **Vordefinierte Analysen**: Häufigste Standorte, Personen-Netzwerke
- **Flexible Filterung**: Nach Zeit, Standort, Personen, Qualität

## Graph-Schema

### Knoten-Typen (Nodes)

```
Image
├── id: Eindeutige Bild-ID
├── name: Dateiname/Pfad
└── Metadaten: EXIF, Qualität, etc.

Person
├── name: Personenname
└── Identifikation: Gesichtserkennung

Face
├── bbox: Gesichtsbereich
├── prob: Erkennungswahrscheinlichkeit
├── emotion: Emotion (happy, neutral, unknown)
├── quality_score: Gesichtsqualität
└── pose: Kopfhaltung (yaw, pitch, roll)

Location
├── lat: Breitengrad
├── lon: Längengrad
└── altitude: Höhe

Address
├── full_address: Vollständige Adresse
├── country: Land
├── state: Bundesland/Staat
├── city: Stadt
└── postcode: Postleitzahl

Camera
├── make: Hersteller
├── model: Modell
└── lens: Objektiv

Tech
├── focal_length: Brennweite
├── f_number: Blende
├── iso: ISO-Wert
└── exposure_time: Belichtungszeit

Capture
├── datetime: Zeitstempel
├── hour: Stunde
├── weekday: Wochentag
└── part_of_day: Tageszeit

ImageAnalysis
├── overall_quality: Gesamtqualität
├── sharpness: Schärfe
├── brightness: Helligkeit
├── contrast: Kontrast
├── noise_level: Rauschpegel
├── aspect_ratio: Seitenverhältnis
└── color_temperature: Farbtemperatur

Dance
└── label: Tanzstil (ballet, hip hop, etc.)
```

### Beziehungen (Relationships)

```
Image -[:CONTAINS]-> Face
Face -[:IDENTIFIED_AS]-> Person
Image -[:AT_LOCATION]-> Location
Image -[:TAKEN_AT]-> Capture
Image -[:HAS_CAMERA]-> Camera
Image -[:HAS_TECH]-> Tech
Image -[:HAS_ANALYSIS]-> ImageAnalysis
Location -[:RESOLVED_AS]-> Address
Face -[:PERFORMS]-> Dance
```

## Verwendung

### 1. Neo4j-Instanz starten

```bash
# Mit Neo4j Desktop
# 1. Neo4j Desktop installieren
# 2. Neue Datenbank erstellen
# 3. Starten und Passwort setzen

# Mit Docker
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 2. Verbindung in Streamlit

1. Öffnen Sie die **Neo4j Graph**-Seite
2. Geben Sie die Verbindungsdaten ein:
   - **URI**: `bolt://localhost:7687`
   - **Benutzername**: `neo4j`
   - **Passwort**: Ihr Neo4j-Passwort
3. Klicken Sie auf **Verbinden**

### 3. Daten importieren

1. Wechseln Sie zum **Import**-Tab
2. Laden Sie eine JSON-Datei hoch (aus der Annotate-Seite)
3. Konfigurieren Sie die Import-Optionen
4. Klicken Sie auf **In Neo4j importieren**

### 4. Daten analysieren

#### Cypher-Abfragen

```cypher
# Alle Personen mit Anzahl ihrer Bilder
MATCH (p:Person)<-[:IDENTIFIED_AS]-(f:Face)-[:CONTAINS]-(img:Image)
RETURN p.name, count(DISTINCT img) as bilder
ORDER BY bilder DESC

# Standorte mit den meisten Fotos
MATCH (img:Image)-[:AT_LOCATION]->(loc:Location)
RETURN loc.lat, loc.lon, count(img) as foto_anzahl
ORDER BY foto_anzahl DESC
LIMIT 10

# Personen-Netzwerk: Wer wurde zusammen fotografiert?
MATCH (p1:Person)<-[:IDENTIFIED_AS]-(f1:Face)-[:CONTAINS]-(img:Image)-[:CONTAINS]-(f2:Face)-[:IDENTIFIED_AS]->(p2:Person)
WHERE p1 <> p2
RETURN p1.name, p2.name, count(img) as zusammen_aufgenommen
ORDER BY zusammen_aufgenommen DESC

# Zeitliche Analyse: Fotos pro Stunde
MATCH (img:Image)-[:TAKEN_AT]->(capture:Capture)
WHERE capture.hour IS NOT NULL
RETURN capture.hour, count(img) as anzahl
ORDER BY capture.hour

# Emotionen-Analyse
MATCH (f:Face)-[:IDENTIFIED_AS]->(p:Person)
WHERE f.emotion IS NOT NULL
RETURN p.name, f.emotion, count(f) as anzahl
ORDER BY p.name, anzahl DESC

# Qualitäts-Analyse
MATCH (img:Image)-[:HAS_ANALYSIS]->(analysis:ImageAnalysis)
WHERE analysis.overall_quality IS NOT NULL
RETURN 
  CASE 
    WHEN analysis.overall_quality > 0.8 THEN 'Hoch'
    WHEN analysis.overall_quality > 0.6 THEN 'Mittel'
    ELSE 'Niedrig'
  END as qualitaet,
  count(img) as anzahl
ORDER BY anzahl DESC
```

#### Visualisierungen

- **Netzwerk-Graph**: Zeigt alle Beziehungen zwischen Entitäten
- **Standort-Karte**: Geografische Verteilung der Fotos
- **Personen-Netzwerk**: Fokus auf eine bestimmte Person

## Erweiterte Funktionen

### Export/Import

```python
# Export der gesamten Graphdatenbank
neo4j.export_to_json("backup.json")

# Import von JSON-Daten
with open("annotations.json") as f:
    data = json.load(f)
neo4j.upsert_annotations(data)
```

### Programmgesteuerte Nutzung

```python
from app.graph_persistence import Neo4jPersistence

# Verbindung herstellen
neo4j = Neo4jPersistence("bolt://localhost:7687", "neo4j", "password")

# Constraints initialisieren
neo4j.init_constraints()

# Daten importieren
neo4j.upsert_annotations(annotation_data)

# Abfragen ausführen
results = neo4j.execute_cypher("MATCH (p:Person) RETURN p.name")

# Graph-Daten für Visualisierung
graph_data = neo4j.get_graph_data(limit=100)

# Verbindung schließen
neo4j.close()
```

## Performance-Optimierung

### Indizes

Die folgenden Constraints/Indizes werden automatisch erstellt:

```cypher
CREATE CONSTRAINT image_id FOR (n:Image) REQUIRE n.id IS UNIQUE
CREATE CONSTRAINT person_name FOR (n:Person) REQUIRE n.name IS UNIQUE
CREATE CONSTRAINT dance_label FOR (n:Dance) REQUIRE n.label IS UNIQUE
```

### Empfohlene Hardware

- **RAM**: Mindestens 4GB für kleine Datensätze, 8GB+ für große
- **CPU**: Multi-Core für bessere Abfrage-Performance
- **Storage**: SSD empfohlen für bessere I/O-Performance

## Troubleshooting

### Verbindungsprobleme

1. **"Connection refused"**: Neo4j-Instanz nicht gestartet
2. **"Authentication failed"**: Falsche Anmeldedaten
3. **"Database not found"**: Datenbankname prüfen

### Performance-Probleme

1. **Langsame Abfragen**: Limit für Visualisierung reduzieren
2. **Speicher-Probleme**: Neo4j-Heap-Größe erhöhen
3. **Timeout-Fehler**: Abfrage-Komplexität reduzieren

### Datenqualität

1. **Fehlende Beziehungen**: JSON-Daten auf Vollständigkeit prüfen
2. **Doppelte Knoten**: MERGE-Statements verwenden
3. **Ungültige Koordinaten**: GPS-Daten validieren

## Beispiele

### Beispiel 1: Personen-Netzwerk analysieren

```cypher
// Finde Personen, die häufig zusammen fotografiert werden
MATCH (p1:Person)<-[:IDENTIFIED_AS]-(f1:Face)-[:CONTAINS]-(img:Image)-[:CONTAINS]-(f2:Face)-[:IDENTIFIED_AS]->(p2:Person)
WHERE p1 <> p2
WITH p1, p2, count(img) as zusammen_anzahl
WHERE zusammen_anzahl >= 3
RETURN p1.name, p2.name, zusammen_anzahl
ORDER BY zusammen_anzahl DESC
```

### Beispiel 2: Standort-basierte Analyse

```cypher
// Finde die beliebtesten Foto-Standorte
MATCH (img:Image)-[:AT_LOCATION]->(loc:Location)
MATCH (img)-[:CONTAINS]->(f:Face)-[:IDENTIFIED_AS]->(p:Person)
RETURN loc.lat, loc.lon, 
       count(DISTINCT img) as foto_anzahl,
       count(DISTINCT p) as personen_anzahl
ORDER BY foto_anzahl DESC
LIMIT 20
```

### Beispiel 3: Zeitliche Trends

```cypher
// Analysiere Foto-Aktivität nach Tageszeit
MATCH (img:Image)-[:TAKEN_AT]->(capture:Capture)
WHERE capture.hour IS NOT NULL
RETURN capture.hour, count(img) as anzahl
ORDER BY capture.hour
```

## Fazit

Die Neo4j-Integration bietet eine mächtige Möglichkeit, Foto-Metadaten zu analysieren und die Beziehungen zwischen verschiedenen Entitäten zu verstehen. Durch die graphische Darstellung können komplexe Zusammenhänge einfach visualisiert und analysiert werden.
