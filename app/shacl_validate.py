from rdflib import Graph
from pyshacl import validate


def validate_rdf(rdf_bytes: bytes, rdf_format: str = "turtle", shapes_path: str = "docs/SHACL/zeitkalkuel_shapes.ttl"):
    data_graph = Graph().parse(data=rdf_bytes, format=rdf_format)
    shapes_graph = Graph().parse(shapes_path, format="turtle")
    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=None,
        inference="rdfs",
        abort_on_first=False,
        meta_shacl=False,
        advanced=False,
        inplace=True,
    )
    return conforms, results_text


