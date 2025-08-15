
from __future__ import annotations
import argparse, os, json
from typing import List, Dict, Any
import cv2
from tqdm import tqdm

from app.face_recognizer import FaceEngine, GalleryDB, build_gallery_from_folder
from app.location import extract_exif_gps, reverse_geocode

def collect_images(path: str, recursive: bool=False) -> List[str]:
    exts = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
    if os.path.isdir(path):
        if recursive:
            res = []
            for root, _, files in os.walk(path):
                for fn in files:
                    if fn.lower().endswith(exts):
                        res.append(os.path.join(root, fn))
            return sorted(res)
        else:
            return sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)])
    else:
        return [path]

def cmd_enroll(args):
    db = build_gallery_from_folder(args.gallery, det_size=(args.det, args.det))
    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    db.save(args.db)
    print(f"Saved gallery DB with {len(db.people)} identities to {args.db}")

def cmd_annotate(args):
    engine = FaceEngine(det_size=(args.det, args.det))
    db = GalleryDB.load(args.db) if args.db and os.path.exists(args.db) else None
    images = collect_images(args.input, recursive=args.recursive)
    out_records: List[Dict[str, Any]] = []
    for path in tqdm(images, desc="Annotating"):
        img = cv2.imread(path)
        if img is None:
            continue
        faces = engine.analyze(img)
        persons = []
        for f in faces:
            name, sim = (None, None)
            if db:
                n, s = db.match(f["embedding"], threshold=args.threshold)
                name, sim = (n, s)
            persons.append({
                "bbox": f["bbox"],
                "prob": f["prob"],
                "name": name,
                "similarity": sim,
                "age": f["age"],
                "gender": f["gender"]
            })
        loc = extract_exif_gps(path)
        addr = reverse_geocode(loc["lat"], loc["lon"]) if (loc and args.reverse_geocode) else None
        record = {
            "image": path,
            "location": {**loc, "address": addr} if loc else None,
            "persons": persons
        }
        out_records.append(record)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    print(f"Wrote annotations for {len(out_records)} images to {args.out}")

def build_parser():
    p = argparse.ArgumentParser(description="Photo metadata annotator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_enroll = sub.add_parser("enroll", help="Build face embedding database from gallery")
    p_enroll.add_argument("--gallery", required=True, help="Path to labeled gallery folder")
    p_enroll.add_argument("--db", required=True, help="Output path to embeddings DB (pickle)")
    p_enroll.add_argument("--det", type=int, default=640, help="Detector size (square)")
    p_enroll.set_defaults(func=cmd_enroll)

    p_annot = sub.add_parser("annotate", help="Annotate photos with faces, age/gender, and GPS location")
    p_annot.add_argument("--input", required=True, help="Image file or folder")
    p_annot.add_argument("--db", required=False, help="Path to embeddings DB (pickle)")
    p_annot.add_argument("--out", required=True, help="Output JSON file")
    p_annot.add_argument("--recursive", action="store_true", help="Recurse into subfolders if input is a directory")
    p_annot.add_argument("--reverse-geocode", action="store_true", help="Convert GPS to address (internet required)")
    p_annot.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity threshold for identity match")
    p_annot.add_argument("--det", type=int, default=640, help="Detector size (square)")
    p_annot.set_defaults(func=cmd_annotate)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
