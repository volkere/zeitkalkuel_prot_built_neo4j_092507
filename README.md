
Zeitkalkül agent_prot_092507


Erweiterte Foto-Metadaten-Analyse mit Gesichtserkennung, EXIF-Extraktion und intelligenten Analysen.  
(Enthält eine CLI und eine Streamlit Multi-Page UI (Enroll + Annotate + Analyze)).

Erweiterte Metadaten-Extraktion
- Vollständige EXIF-Daten: Kamera-Modell, Objektiv, Aufnahme-Einstellungen
- GPS mit Höhenangabe: Präzise Standortdaten mit Altitude
- Detaillierte Standort-Info: Vollständige Adressen und geografische Details
- Zeitstempel-Parsing: Unterstützt verschiedene Datumsformate

Verbesserte Gesichtserkennung
- Qualitätsbewertung: Automatische Bewertung der Gesichtsqualität
- Emotions-Erkennung: Happy, neutral, unknown
- Status-Erkennung: Augen (offen/geschlossen) und Mund-Status
- Pose-Schätzung: Yaw, Pitch, Roll-Winkel
- Erweiterte Demografie: Alters- und Geschlechtserkennung

Intelligente Analyse
- Interaktive Visualisierungen: Charts und Statistiken mit Plotly
- Automatische Gruppierung: Nach Standort und Zeit
- Qualitätsfilter: Filtert nach Gesichtsqualität und -größe
- Export-Funktionen: JSON-Export für weitere Verarbeitung

Features
- Face detection & embeddings (InsightFace `buffalo_l`)
- Age & gender estimation (approximate)
- Known-person matching via embeddings database (`embeddings.pkl`)
- EXIF GPS extraction with optional reverse geocoding
- Erweiterte Metadaten-Extraktion (Kamera, Einstellungen, Datum)
- Qualitätsbewertung für Gesichter und Bilder
- Emotions- und Status-Erkennung
- Interaktive Analysen mit Charts und Statistiken
- Intelligente Bildgruppierung
- Streamlit UI mit drag & drop, bounding boxes, JSON export
- CLI for batch processing
 - CLIP-basierte Bild-Embeddings und semantische Suche mit FAISS
 - Query-by-Image und Query-by-Text für Bildsuche
 - Zero-shot Dance-Erkennung pro Gesicht (konfigurierbare Labels)
 - Zeit-Anreicherung: Wochentag, Stunde, Tagesteil
 - Karten-Links in Metadaten (OpenStreetMap, Google Maps)
 - Bildanalyse: Qualitätsmetriken, Farbhistogramme, Kompositionsmerkmale
 - ZIP-Export der annotierten Vorschaubilder

> Use responsibly: Face analysis and attribute inference can be biased and regulated. Ensure you have the right to process the images and comply with local laws (see `docs/PRIVACY.md`).

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
- Enroll: Erstellen von Embeddings für Personen-Erkennung
- Annotate: Erweiterte Foto-Analyse mit Metadaten
- Analyze: Statistiken, Charts und Gruppierungsanalyse
 - Image Search: Semantische Bildsuche mit CLIP-Embeddings (Bild- oder Text-Query)
 - Neo4j Graph: Graphdatenbank-Integration und -Visualisierung (Import/Abfragen/Export)
 - Linked Open Data Export: RDF-Export (Turtle/JSON-LD)

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
pages/                # Streamlit pages (Enroll, Annotate, Analyze, Image Search, Neo4j, LOD Export)
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
- Gesichtsqualität: Filtert nach Schärfe, Helligkeit, Kontrast
- Größenfilter: Mindestgröße für Gesichter
- Qualitätsbewertung: Automatische Bewertung von 0-1

2. Erweiterte EXIF-Parsing
- Mehr Formate: Unterstützt verschiedene EXIF-Standards
- Vollständige Metadaten: Kamera, Objektiv, Einstellungen
- Fehlerbehandlung: Robuste Parsing-Logik

3. Intelligente Gruppierung
- Standort-Gruppierung: Gruppiert Bilder in 100m-Radius
- Zeit-Gruppierung: Gruppiert nach 24h-Zeitfenster
- Ähnlichkeitsanalyse: Automatische Kategorisierung

4. Visualisierungen
- Interaktive Charts: Plotly-basierte Visualisierungen
- Statistiken: Alters-, Qualitäts-, Kamera-Verteilungen
- Karten: GPS-Standorte auf interaktiven Karten

5. Export-Funktionen
- JSON-Export: Vollständige Metadaten
- Analyse-Export: Gruppierungen und Statistiken
- Format-Kompatibilität: Standardisierte Ausgabe


Neo4j-Integration (optional)
----------------------------
Die Seite "Neo4j Graph" erlaubt es, Annotationen in eine Neo4j-Instanz zu importieren, Abfragen auszuführen und Graphen zu visualisieren.

Kurzüberblick:
- Verbindung: Bolt-URI, Benutzer, Passwort in der Sidebar angeben und verbinden
- Import: JSON-Ergebnisse (aus "Annotate") in die DB importieren
- Abfragen: Eigene Cypher-Queries ausführen
- Visualisierung: Interaktives Netzwerk (pyvis) oder statisch (networkx)
- Verwaltung: Export nach JSON, Datenbank leeren

Weitere Details: siehe `docs/NEO4J_INTEGRATION.md`.


Erweiterte Graph-Visualisierung
-------------------------------
Die interaktive Netzwerkansicht (pyvis) auf der Seite "Neo4j Graph" bietet erweiterte Steuerungsmöglichkeiten.

Features:
- Steuerpanel (optional): Physics, Nodes, Edges
- Physik ein-/ausschalten (Force-Layout)
- Knotengröße nach Degree-Zentralität skalieren
- Filter für minimalen Knotengrad (blendet schwach vernetzte Knoten aus)
- Optional Kanten-Labels rendern
- Farbgebung und Tooltips nach Labels (z. B. Person, Image, Location, Face)

Bedienung (Tab "Visualisierung"):
- Limit: maximale Anzahl geladener Knoten/Kanten
- Steuerung anzeigen: blendet das pyvis-Control-Panel ein
- Physik aktiv: Layout-Dynamik ein/aus
- Größe nach Zentralität: skaliert Knoten nach Degree-Centrality
- Minimaler Knotengrad: filtert Knoten unterhalb des Grenzwerts
- Kanten-Labels anzeigen: zeigt Beschriftungen an Kanten

Hinweis: Standardhöhe ist 700px, für große Graphen empfiehlt sich die aktive Physik und ggf. Erhöhung des Limits.


Linked Open Data (LOD) Export
-----------------------------
Die Seite "Linked Open Data Export" exportiert die Neo4j-Daten als RDF in den Formaten Turtle oder JSON‑LD.

Verwendung:
- Seite öffnen: "Linked Open Data Export"
- Verbindung zur Neo4j-DB angeben (Bolt URI, User, Passwort)
- Base-URI konfigurieren (Standard lokal: `http://localhost:8000/zeitkalkuel/`)
- Format wählen (`turtle` oder `json-ld`) und Export starten
- Ergebnisdatei herunterladen

Ontologie-Mapping (vereinfachte Zuordnung):
- Image → `schema:ImageObject`
- Person → `foaf:Person`
- Location → `schema:Place` mit `geo:lat`/`geo:long`
- Capture → `schema:Event` (Zeitstempel)
- Address → `schema:PostalAddress`
- Kanten werden – wenn verfügbar – auf `schema:*` gemappt, sonst auf `zk:*` (Projekt-Namespace)

Abhängigkeit: `rdflib` (bereits in `requirements.txt`).


LOD API (FastAPI)
-----------------
Zusätzlich zur Streamlit-Seite steht eine HTTP‑API zur Verfügung, die RDF on‑the‑fly erzeugt und SHACL‑Validierung anbietet.

Starten:
```bash
uvicorn app.lod_api:app --host 0.0.0.0 --port 8000 --reload
```

Endpunkte:
- RDF Export: `GET /rdf?uri=bolt://localhost:7687&user=neo4j&password=...&base_uri=https://example.org/zeitkalkuel/&fmt=turtle`
  - `fmt`: `turtle` | `json-ld` | `nt`
- SHACL Validation: `POST /validate?uri=...&user=...&password=...&base_uri=...&rdf_format=turtle`

Beispiele:
```bash
curl "http://localhost:8000/rdf?fmt=turtle" -o graph.ttl
curl -X POST "http://localhost:8000/validate?rdf_format=turtle" -i
```

Ontologie-Mappings
------------------
Die RDF-Serialisierung nutzt folgende Vokabulare:
- `schema.org` (z. B. `schema:ImageObject`, `schema:Place`, `schema:Event`, `schema:PostalAddress`)
- `wgs84_pos` (`geo:lat`, `geo:long`)
- `FOAF` (`foaf:Person`, `foaf:name`)
- `EXIF` (z. B. `exif:make`, `exif:model`, `exif:lens`, `exif:dateTimeOriginal` – Best‑Effort aus vorhandenen Attributen)
- `IPTC` (Platzhalter für Foto-Metadaten, Best‑Effort)
- `PROV-O` (`prov:generatedAtTime` für Aufnahmereignisse)

Projektinterne Ontologie
------------------------
Datei: `docs/ontology/zeitkalkuel.ttl` (Namespace `zk:`)
- Zusätzliche Klassen: `zk:Tech`, `zk:ImageAnalysis`, `zk:Dance`
- Objekt‑Properties: `zk:hasTech`, `zk:hasAnalysis`, `zk:performs`
- Daten‑Properties: `zk:overallQuality`, `zk:sharpness`, `zk:brightness`, `zk:contrast`, `zk:noiseLevel`, `zk:aspectRatio`, `zk:colorTemperature`, `zk:similarity`
Diese ergänzen schema.org/EXIF/FOAF und werden im RDF‑Export als Fallback bzw. für Analysewerte genutzt.

SHACL‑Validierung
-----------------
Shapes-Datei: `docs/SHACL/zeitkalkuel_shapes.ttl` (Image mit Titel, Person mit Name, Place mit lat/long).

Programmatisch:
```python
from app.shacl_validate import validate_rdf
conforms, report = validate_rdf(open("graph.ttl","rb").read(), rdf_format="turtle")
print(conforms); print(report)
```

Dereferenzierbare URIs
----------------------
Setzen Sie im Export/API `base_uri` zunächst lokal auf `http://localhost:8000/zeitkalkuel/`. Später können Sie auf eine eigene Domain wechseln (z. B. `https://data.meine-domain.de/zeitkalkuel/`).

Hinweise:
- Verwenden Sie einen Reverse‑Proxy (z. B. NGINX/Traefik), der `https://data.meine-domain.de/rdf?...` auf die FastAPI‑Instanz weiterleitet.
- Bewahren Sie stabile Bezeichner (z. B. Hash des Bildes) für Ressource‑URIs.
- Für Content‑Negotiation per Query‑Param `fmt` (`turtle|json-ld|nt`).


