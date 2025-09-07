from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from neo4j import GraphDatabase, Driver
import json
import hashlib
from datetime import datetime


class Neo4jPersistence:
    """Persistence layer for storing annotation JSON into Neo4j.

    Schema (labels and relationships):
    - (Image {id, name, path, width, height})
      -[:TAKEN_AT]-> (Capture {datetime, hour, weekday, part_of_day})
      -[:AT_LOCATION]-> (Location {lat, lon, altitude})
      -[:HAS_CAMERA]-> (Camera {make, model, lens})
      -[:HAS_TECH]-> (Tech {focal_length, f_number, iso, exposure_time})
      -[:HAS_ANALYSIS]-> (ImageAnalysis {overall_quality, sharpness, brightness, contrast, noise_level, aspect_ratio, color_temperature})
      -[:CONTAINS]-> (Face {bbox, prob, quality_score, emotion, eye_status, mouth_status, pose})
    - (Face)-[:IDENTIFIED_AS]->(Person {name}) with similarity
    - (Location)-[:RESOLVED_AS]->(Address {full_address, country, state, city, postcode})
    - (Face)-[:PERFORMS]->(Dance {label}) with score
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._uri = uri
        self._user = user

    def close(self):
        self._driver.close()

    def init_constraints(self):
        """Create indexes/constraints for faster upserts."""
        cypher = [
            "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (n:Image) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (n:Person) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT dance_label IF NOT EXISTS FOR (n:Dance) REQUIRE n.label IS UNIQUE",
        ]
        with self._driver.session() as session:
            for stmt in cypher:
                session.run(stmt)

    def upsert_annotations(self, records: List[Dict[str, Any]]):
        with self._driver.session() as session:
            for rec in records:
                self._upsert_one(session, rec)

    def _upsert_one(self, session, rec: Dict[str, Any]):
        image_name = rec.get("image")
        image_id = image_name  # can be replaced by hash if needed
        metadata = rec.get("metadata") or {}
        location = rec.get("location") or {}
        persons = rec.get("persons") or []
        analysis = (rec.get("image_analysis") or {}).get("quality") or {}
        composition = (rec.get("image_analysis") or {}).get("composition") or {}

        # Image node
        session.run(
            "MERGE (img:Image {id:$id}) SET img.name=$name",
            {"id": image_id, "name": image_name}
        )

        # Capture temporal metadata
        dt = metadata.get("datetime")
        time_meta = metadata.get("time") or {}
        if dt or time_meta:
            session.run(
                """
                MERGE (c:Capture {id:$id})
                SET c.datetime=$datetime, c.hour=$hour, c.weekday=$weekday, c.part_of_day=$part_of_day
                WITH c
                MATCH (img:Image {id:$id})
                MERGE (img)-[:TAKEN_AT]->(c)
                """,
                {
                    "id": image_id,
                    "datetime": dt,
                    "hour": time_meta.get("hour"),
                    "weekday": time_meta.get("weekday"),
                    "part_of_day": time_meta.get("part_of_day"),
                },
            )

        # Camera and technical
        cam_props = {k: metadata.get(k) for k in ["camera_make", "camera_model", "lens"]}
        if any(cam_props.values()):
            session.run(
                """
                MERGE (cam:Camera {make:$make, model:$model, lens:$lens})
                WITH cam
                MATCH (img:Image {id:$id})
                MERGE (img)-[:HAS_CAMERA]->(cam)
                """,
                {"id": image_id, "make": cam_props.get("camera_make"), "model": cam_props.get("camera_model"), "lens": cam_props.get("lens")},
            )

        tech_props = {k: metadata.get(k) for k in ["focal_length", "f_number", "iso", "exposure_time"]}
        if any(tech_props.values()):
            session.run(
                """
                MERGE (t:Tech {id:$id})
                SET t.focal_length=$focal_length, t.f_number=$f_number, t.iso=$iso, t.exposure_time=$exposure_time
                WITH t
                MATCH (img:Image {id:$id})
                MERGE (img)-[:HAS_TECH]->(t)
                """,
                {"id": image_id, **tech_props},
            )

        # Location and address
        gps = metadata.get("gps") or location or {}
        if gps.get("lat") is not None and gps.get("lon") is not None:
            session.run(
                """
                MERGE (loc:Location {lat:$lat, lon:$lon})
                SET loc.altitude=$altitude
                WITH loc
                MATCH (img:Image {id:$id})
                MERGE (img)-[:AT_LOCATION]->(loc)
                """,
                {"id": image_id, "lat": gps.get("lat"), "lon": gps.get("lon"), "altitude": gps.get("altitude")},
            )
            addr = rec.get("location") or {}
            if addr.get("full_address"):
                session.run(
                    """
                    MERGE (a:Address {full_address:$full})
                    SET a.country=$country, a.state=$state, a.city=$city, a.postcode=$postcode
                    WITH a
                    MATCH (loc:Location {lat:$lat, lon:$lon})
                    MERGE (loc)-[:RESOLVED_AS]->(a)
                    """,
                    {
                        "full": addr.get("full_address"),
                        "country": addr.get("country"),
                        "state": addr.get("state"),
                        "city": addr.get("city"),
                        "postcode": addr.get("postcode"),
                        "lat": gps.get("lat"),
                        "lon": gps.get("lon"),
                    },
                )

        # Image analysis
        if analysis or composition:
            session.run(
                """
                MERGE (ia:ImageAnalysis {id:$id})
                SET ia.overall_quality=$overall_quality, ia.sharpness=$sharpness, ia.brightness=$brightness,
                    ia.contrast=$contrast, ia.noise_level=$noise_level, ia.aspect_ratio=$aspect_ratio,
                    ia.color_temperature=$color_temperature
                WITH ia
                MATCH (img:Image {id:$id})
                MERGE (img)-[:HAS_ANALYSIS]->(ia)
                """,
                {
                    "id": image_id,
                    "overall_quality": analysis.get("overall_quality"),
                    "sharpness": analysis.get("sharpness"),
                    "brightness": analysis.get("brightness"),
                    "contrast": analysis.get("contrast"),
                    "noise_level": analysis.get("noise_level"),
                    "aspect_ratio": composition.get("aspect_ratio"),
                    "color_temperature": composition.get("color_temperature"),
                },
            )

        # Faces and persons
        for idx, p in enumerate(persons):
            bbox = p.get("bbox") or [0, 0, 0, 0]
            face_id = f"{image_id}#face{idx}"
            session.run(
                """
                MERGE (f:Face {id:$fid})
                SET f.bbox=$bbox, f.prob=$prob, f.quality_score=$quality_score, f.emotion=$emotion,
                    f.eye_status=$eye_status, f.mouth_status=$mouth_status, f.similarity=$similarity
                WITH f
                MATCH (img:Image {id:$iid})
                MERGE (img)-[:CONTAINS]->(f)
                """,
                {
                    "fid": face_id,
                    "bbox": bbox,
                    "prob": p.get("prob"),
                    "quality_score": p.get("quality_score"),
                    "emotion": p.get("emotion"),
                    "eye_status": p.get("eye_status"),
                    "mouth_status": p.get("mouth_status"),
                    "similarity": p.get("similarity"),
                    "iid": image_id,
                },
            )
            # Pose if available
            pose = p.get("pose")
            if pose:
                session.run(
                    "MATCH (f:Face {id:$fid}) SET f.pose_yaw=$yaw, f.pose_pitch=$pitch, f.pose_roll=$roll",
                    {"fid": face_id, "yaw": pose.get("yaw"), "pitch": pose.get("pitch"), "roll": pose.get("roll")},
                )

            # Identified person
            if p.get("name"):
                session.run(
                    """
                    MERGE (per:Person {name:$name})
                    WITH per
                    MATCH (f:Face {id:$fid})
                    MERGE (f)-[r:IDENTIFIED_AS]->(per)
                    SET r.similarity=$sim
                    """,
                    {"name": p["name"], "fid": face_id, "sim": p.get("similarity")},
                )

            # Dance
            if p.get("dance"):
                session.run(
                    """
                    MERGE (d:Dance {label:$label})
                    WITH d
                    MATCH (f:Face {id:$fid})
                    MERGE (f)-[r:PERFORMS]->(d)
                    SET r.score=$score
                    """,
                    {"fid": face_id, "label": p["dance"], "score": p.get("dance_score")},
                )

    def test_connection(self) -> bool:
        """Test if the Neo4j connection is working."""
        try:
            with self._driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception:
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the Neo4j database."""
        try:
            with self._driver.session() as session:
                # Get node counts
                node_counts = {}
                result = session.run("CALL db.labels()")
                for record in result:
                    label = record["label"]
                    count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    node_counts[label] = count_result.single()["count"]
                
                # Get relationship counts
                rel_counts = {}
                result = session.run("CALL db.relationshipTypes()")
                for record in result:
                    rel_type = record["relationshipType"]
                    count_result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                    rel_counts[rel_type] = count_result.single()["count"]
                
                return {
                    "node_counts": node_counts,
                    "relationship_counts": rel_counts,
                    "uri": self._uri,
                    "user": self._user
                }
        except Exception as e:
            return {"error": str(e)}

    def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            return [{"error": str(e)}]

    def get_graph_data(self, limit: int = 100) -> Dict[str, Any]:
        """Get graph data for visualization."""
        try:
            with self._driver.session() as session:
                # Get nodes
                nodes_query = """
                MATCH (n)
                RETURN n, labels(n) as labels, id(n) as id
                LIMIT $limit
                """
                nodes_result = session.run(nodes_query, {"limit": limit})
                nodes = []
                for record in nodes_result:
                    node_data = dict(record["n"])
                    nodes.append({
                        "id": record["id"],
                        "labels": record["labels"],
                        "properties": node_data
                    })
                
                # Get relationships
                rels_query = """
                MATCH (a)-[r]->(b)
                RETURN id(a) as source, id(b) as target, type(r) as type, properties(r) as properties
                LIMIT $limit
                """
                rels_result = session.run(rels_query, {"limit": limit})
                relationships = []
                for record in rels_result:
                    relationships.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "properties": dict(record["properties"])
                    })
                
                return {
                    "nodes": nodes,
                    "relationships": relationships
                }
        except Exception as e:
            return {"error": str(e)}

    def get_person_network(self, person_name: str) -> Dict[str, Any]:
        """Get the network of a specific person."""
        try:
            with self._driver.session() as session:
                query = """
                MATCH (p:Person {name: $name})
                OPTIONAL MATCH (p)<-[:IDENTIFIED_AS]-(f:Face)-[:CONTAINS]-(img:Image)
                OPTIONAL MATCH (img)-[:AT_LOCATION]->(loc:Location)
                OPTIONAL MATCH (img)-[:TAKEN_AT]->(capture:Capture)
                RETURN p, collect(DISTINCT f) as faces, 
                       collect(DISTINCT img) as images,
                       collect(DISTINCT loc) as locations,
                       collect(DISTINCT capture) as captures
                """
                result = session.run(query, {"name": person_name})
                record = result.single()
                if record:
                    return {
                        "person": dict(record["p"]),
                        "faces": [dict(f) for f in record["faces"] if f],
                        "images": [dict(img) for img in record["images"] if img],
                        "locations": [dict(loc) for loc in record["locations"] if loc],
                        "captures": [dict(cap) for cap in record["captures"] if cap]
                    }
                return {}
        except Exception as e:
            return {"error": str(e)}

    def get_location_analysis(self) -> Dict[str, Any]:
        """Get analysis of photo locations."""
        try:
            with self._driver.session() as session:
                # Most photographed locations
                query = """
                MATCH (img:Image)-[:AT_LOCATION]->(loc:Location)
                RETURN loc.lat as lat, loc.lon as lon, count(img) as photo_count
                ORDER BY photo_count DESC
                LIMIT 20
                """
                result = session.run(query)
                locations = [dict(record) for record in result]
                
                # Location with most people
                query2 = """
                MATCH (img:Image)-[:AT_LOCATION]->(loc:Location)
                MATCH (img)-[:CONTAINS]->(f:Face)-[:IDENTIFIED_AS]->(p:Person)
                RETURN loc.lat as lat, loc.lon as lon, 
                       count(DISTINCT p) as unique_people,
                       count(DISTINCT img) as photo_count
                ORDER BY unique_people DESC
                LIMIT 10
                """
                result2 = session.run(query2)
                people_locations = [dict(record) for record in result2]
                
                return {
                    "most_photographed": locations,
                    "most_people": people_locations
                }
        except Exception as e:
            return {"error": str(e)}

    def clear_database(self) -> bool:
        """Clear all data from the database."""
        try:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                return True
        except Exception as e:
            return False

    def export_to_json(self, output_file: str) -> bool:
        """Export all graph data to JSON file."""
        try:
            graph_data = self.get_graph_data(limit=10000)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            return False



