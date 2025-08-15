
# Architektur

- **insightface** (buffalo_l) für Detection, Landmarks, Embeddings, Age/Gender.
- **OpenCV** für Bild-I/O/Zeichnen.
- **Pillow**/EXIF für GPS (EXIF), optional **geopy** für Reverse-Geocoding.
- **Streamlit** als Multi-Page UI (Enroll & Annotate).
- **CLI** (argparse) für Batchläufe.
