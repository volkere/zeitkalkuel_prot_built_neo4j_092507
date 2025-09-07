import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image


@st.cache_resource(show_spinner=False)
def load_clip_model():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor, device


def read_image_paths(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() in exts:
        return [str(root_path)]
    if not root_path.exists():
        return []
    paths = [str(p) for p in root_path.rglob("*") if p.suffix.lower() in exts]
    return sorted(paths)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def compute_image_embeddings(paths: List[str], batch_size: int = 16) -> np.ndarray:
    import torch

    model, processor, device = load_clip_model()
    vectors: List[np.ndarray] = []
    for batch in chunk_list(paths, batch_size):
        images = [load_image(p) for p in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)  # [B, d]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.detach().cpu().numpy().astype("float32"))
    if not vectors:
        return np.zeros((0, 512), dtype="float32")
    return np.vstack(vectors)


@st.cache_resource(show_spinner=False)
def build_faiss_index(embeddings: np.ndarray):
    import faiss

    if embeddings.size == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search_index(index, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if index is None or query_vec.size == 0:
        return np.array([]), np.array([])
    D, I = index.search(query_vec, k)
    return D[0], I[0]


def app():
    st.title("Image Embedding Search (CLIP)")
    st.caption("Suche ähnliche Bilder per semantischer Ähnlichkeit")

    with st.sidebar:
        st.header("Index konfigurieren")
        root_dir = st.text_input("Bildordner oder Bilddatei", value=str(Path.cwd()))
        batch_size = st.slider("Batch-Größe", 4, 64, 16, 4)
        k = st.slider("Top-K Ergebnisse", 1, 20, 5)
        build_button = st.button("Index neu aufbauen")

    if "image_paths" not in st.session_state:
        st.session_state.image_paths = []
        st.session_state.embeddings = None
        st.session_state.index = None

    if build_button:
        with st.spinner("Lese Bilder und berechne Embeddings..."):
            paths = read_image_paths(root_dir)
            st.session_state.image_paths = paths
            if len(paths) == 0:
                st.warning("Keine Bilder gefunden.")
            else:
                start = time.time()
                embs = compute_image_embeddings(paths, batch_size=batch_size)
                st.session_state.embeddings = embs
                st.session_state.index = build_faiss_index(embs)
                elapsed = time.time() - start
                st.success(f"Index mit {len(paths)} Bildern in {elapsed:.1f}s erstellt.")

    st.subheader("Query")
    col_upload, col_or, col_text = st.columns([3, 1, 3])
    with col_upload:
        uploaded = st.file_uploader("Query-Bild hochladen", type=["jpg", "jpeg", "png", "webp", "bmp"])
    with col_or:
        st.write("\n\noder")
    with col_text:
        text_query = st.text_input("oder Text-Query (optional)")

    query_vec = None
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query", use_column_width=True)
        import torch
        _, processor, device = load_clip_model()
        inputs = processor(images=img, return_tensors="pt").to(device)
        from transformers import CLIPModel
        model, _, _ = load_clip_model()
        with torch.no_grad():
            q = model.get_image_features(**inputs)
            q = q / q.norm(dim=-1, keepdim=True)
        query_vec = q.cpu().numpy().astype("float32")
    elif text_query:
        import torch
        model, processor, device = load_clip_model()
        inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            q = model.get_text_features(**inputs)
            q = q / q.norm(dim=-1, keepdim=True)
        query_vec = q.cpu().numpy().astype("float32")

    st.divider()
    st.subheader("Ergebnisse")

    if query_vec is not None and st.session_state.index is not None and st.session_state.image_paths:
        scores, ids = search_index(st.session_state.index, query_vec, k)
        cols = st.columns(min(k, 5))
        for rank, (idx, score) in enumerate(zip(ids, scores)):
            if idx < 0 or idx >= len(st.session_state.image_paths):
                continue
            path = st.session_state.image_paths[int(idx)]
            with cols[rank % len(cols)]:
                st.image(path, caption=f"{Path(path).name} | cos={float(score):.3f}")
    else:
        st.info("Baue zuerst den Index und lade ein Query-Bild oder gib Text ein.")


if __name__ == "__main__":
    app()


