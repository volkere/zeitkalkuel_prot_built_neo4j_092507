
# Nutzung: CLI

Embeddings bauen:
```bash
python -m app.main enroll --gallery ./gallery --db embeddings.pkl
```

Fotos annotieren:
```bash
python -m app.main annotate --input ./photos --out output.json --recursive --reverse-geocode
```
