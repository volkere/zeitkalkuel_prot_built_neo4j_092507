from fastapi import FastAPI, Response, Query
from enum import Enum
import tempfile
from .graph_persistence import Neo4jPersistence
from .shacl_validate import validate_rdf


class RDFFormat(str, Enum):
    turtle = "turtle"
    jsonld = "json-ld"
    ntriples = "nt"


def create_app() -> FastAPI:
    app = FastAPI(title="Zeitkalkül LOD API", version="1.0.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/rdf")
    def export_rdf(
        uri: str = Query("bolt://localhost:7687"),
        user: str = Query("neo4j"),
        password: str = Query(""),
        base_uri: str = Query("https://example.org/zeitkalkuel/"),
        fmt: RDFFormat = Query(RDFFormat.turtle),
    ):
        neo = Neo4jPersistence(uri, user, password)
        tmp = tempfile.mktemp(suffix=(".ttl" if fmt == RDFFormat.turtle else ".jsonld" if fmt == RDFFormat.jsonld else ".nt"))
        ok = neo.export_rdf(tmp, fmt=("json-ld" if fmt == RDFFormat.jsonld else ("nt" if fmt == RDFFormat.ntriples else "turtle")), base_uri=base_uri)
        neo.close()
        if not ok:
            return Response("Export failed", status_code=500)
        with open(tmp, "rb") as f:
            data = f.read()
        media = {
            RDFFormat.turtle: "text/turtle",
            RDFFormat.jsonld: "application/ld+json",
            RDFFormat.ntriples: "application/n-triples",
        }[fmt]
        return Response(content=data, media_type=media)

    @app.post("/validate")
    def validate_endpoint(
        rdf_format: RDFFormat = Query(RDFFormat.turtle),
        base_uri: str = Query("https://example.org/zeitkalkuel/"),
        uri: str = Query("bolt://localhost:7687"),
        user: str = Query("neo4j"),
        password: str = Query(""),
    ):
        # Produce RDF on the fly then validate
        neo = Neo4jPersistence(uri, user, password)
        import tempfile
        tmp = tempfile.mktemp(suffix=(".ttl" if rdf_format == RDFFormat.turtle else ".jsonld"))
        ok = neo.export_rdf(tmp, fmt=("json-ld" if rdf_format == RDFFormat.jsonld else "turtle"), base_uri=base_uri)
        neo.close()
        if not ok:
            return Response("Export failed", status_code=500)
        with open(tmp, "rb") as f:
            data = f.read()
        conforms, results_text = validate_rdf(data, rdf_format=("json-ld" if rdf_format == RDFFormat.jsonld else "turtle"))
        status = 200 if conforms else 422
        return Response(results_text, status_code=status, media_type="text/plain")

    return app


app = create_app()


