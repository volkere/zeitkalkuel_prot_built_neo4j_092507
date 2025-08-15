
import io, os, zipfile, tempfile, pickle
from typing import List, Dict
import streamlit as st
import numpy as np
import cv2

from app.face_recognizer import FaceEngine, GalleryDB

st.title("👤 Enroll: Embeddings erstellen")
st.caption("Erzeuge eine embeddings.pkl aus einer Galerie-ZIP oder manuell pro Person.")

with st.sidebar:
    det = st.slider("Detector size", 320, 1024, 640, 64)

if "engine_enroll" not in st.session_state or st.session_state.get("det_enroll") != det:
    st.session_state["engine_enroll"] = FaceEngine(det_size=(det, det))
    st.session_state["det_enroll"] = det

tab_zip, tab_manual = st.tabs(["📦 Galerie-ZIP hochladen", "🧩 Manuell pro Person"])

with tab_zip:
    st.markdown("**ZIP-Struktur:** `PersonA/*.jpg`, `PersonB/*.png`, …")
    zip_file = st.file_uploader("Galerie-ZIP auswählen", type=["zip"])
    if zip_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            z = zipfile.ZipFile(zip_file)
            z.extractall(tmpdir)
            db = GalleryDB()
            count_imgs = 0
            for root, dirs, files in os.walk(tmpdir):
                for fn in files:
                    if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")):
                        p = os.path.join(root, fn)
                        img = cv2.imread(p)
                        if img is None:
                            continue
                        faces = st.session_state["engine_enroll"].analyze(img)
                        if not faces:
                            continue
                        faces.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
                        person = os.path.basename(os.path.dirname(p))
                        db.add(person, faces[0]["embedding"])
                        count_imgs += 1
            st.success(f"Embeddings erstellt: {len(db.people)} Personen aus {count_imgs} Bildern.")
            b = io.BytesIO()
            pickle.dump(db.people, b, protocol=pickle.HIGHEST_PROTOCOL)
            st.download_button("⬇️ embeddings.pkl herunterladen", data=b.getvalue(), file_name="embeddings.pkl", mime="application/octet-stream")

with tab_manual:
    st.markdown("Name eingeben, Bilder hochladen, **Hinzufügen** klicken.")
    if "manual_db" not in st.session_state:
        st.session_state["manual_db"] = GalleryDB()
    name = st.text_input("Name der Person")
    imgs = st.file_uploader("Bilder der Person", type=["jpg","jpeg","png","bmp","webp","tif","tiff"], accept_multiple_files=True)
    if st.button("Hinzufügen", disabled=(not name or not imgs)):
        added = 0
        for up in imgs:
            data = up.read()
            file_bytes = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue
            faces = st.session_state["engine_enroll"].analyze(img)
            if not faces:
                continue
            faces.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
            st.session_state["manual_db"].add(name, faces[0]["embedding"])
            added += 1
        st.success(f"{added} Bild(er) für **{name}** hinzugefügt.")
    if st.session_state["manual_db"].people:
        st.info(f"Aktueller DB-Status: {len(st.session_state['manual_db'].people)} Personen.")
        b = io.BytesIO()
        pickle.dump(st.session_state["manual_db"].people, b, protocol=pickle.HIGHEST_PROTOCOL)
        st.download_button("⬇️ embeddings.pkl herunterladen", data=b.getvalue(), file_name="embeddings.pkl", mime="application/octet-stream")
