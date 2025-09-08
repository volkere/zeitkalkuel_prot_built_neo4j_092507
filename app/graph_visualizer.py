"""
Enhanced graph visualization utilities for Neo4j data.
"""
from typing import Dict, Any, List, Optional, Union
import networkx as nx
from pyvis.network import Network
import json
import io
import base64


class GraphVisualizer:
    """Enhanced graph visualization for Neo4j data."""
    
    def __init__(self, width: str = "100%", height: str = "600px"):
        self.width = width
        self.height = height
    
    def create_interactive_network(self, graph_data: Dict[str, Any], 
                                 title: str = "Neo4j Graph",
                                 show_buttons: bool = True,
                                 scale_by_centrality: bool = True,
                                 physics: bool = True,
                                 min_degree: int = 0,
                                 show_edge_labels: bool = True,
                                 highlight_nodes: Union[set, None] = None) -> str:
        """Create an interactive network visualization using pyvis.

        - show_buttons: adds a control panel (physics/nodes/edges).
        - scale_by_centrality: node sizes based on degree centrality.
        - physics: enable/disable force layout.
        - min_degree: filter nodes with degree less than threshold.
        - show_edge_labels: render edge labels.
        """
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor="#222222",
            font_color="white",
            directed=True
        )
        
        # Configure physics
        net.set_options("""
        {
            "physics": {
                "enabled": %s,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                }
            }
        }
        """ % ("true" if physics else "false"))

        if show_buttons:
            try:
                net.show_buttons(filter_=["physics", "nodes", "edges"])
            except Exception:
                pass
        
        # Build a quick NetworkX graph for centrality and degree filtering
        G = nx.DiGraph()

        # Add nodes
        node_colors = {
            "Person": "#ff6b6b",
            "Image": "#4ecdc4", 
            "Location": "#45b7d1",
            "Face": "#f9ca24",
            "Camera": "#6c5ce7",
            "Tech": "#a29bfe",
            "Capture": "#fd79a8",
            "ImageAnalysis": "#fdcb6e",
            "Address": "#e17055",
            "Dance": "#00b894"
        }
        
        for node in graph_data.get("nodes", []):
            node_id = node["id"]
            labels = node["labels"]
            properties = node["properties"]
            G.add_node(node_id)
            
            # Determine node color
            color = "#95a5a6"  # default gray
            for label in labels:
                if label in node_colors:
                    color = node_colors[label]
                    break
            
            # Create node label with more descriptive information
            if "Person" in labels:
                name = properties.get("name", "Unknown")
                label = name
                title_text = f"Person: {name}"
            elif "Image" in labels:
                name = properties.get("name", "Unknown")
                label = name[:20] + "..." if len(name) > 20 else name
                title_text = f"Image: {name}"
            elif "Location" in labels:
                lat = properties.get("lat", "?")
                lon = properties.get("lon", "?")
                label = f"📍 {lat:.3f}, {lon:.3f}"
                title_text = f"Location: {lat}, {lon}"
            elif "Face" in labels:
                emotion = properties.get("emotion", "unknown")
                quality = properties.get("quality_score", 0)
                label = f"😊 {emotion}"
                title_text = f"Face: {emotion} (Quality: {quality:.2f})"
            elif "Camera" in labels:
                make = properties.get("make", "Unknown")
                model = properties.get("model", "Unknown")
                label = f"📷 {make} {model}"
                title_text = f"Camera: {make} {model}"
            elif "Tech" in labels:
                focal = properties.get("focal_length", "?")
                fnum = properties.get("f_number", "?")
                label = f"⚙️ {focal}mm f/{fnum}"
                title_text = f"Tech: {focal}mm, f/{fnum}, ISO {properties.get('iso', '?')}"
            elif "Capture" in labels:
                dt = properties.get("datetime", "Unknown")
                label = f"📅 {dt[:10] if dt != 'Unknown' else 'Unknown'}"
                title_text = f"Capture: {dt}"
            elif "ImageAnalysis" in labels:
                quality = properties.get("overall_quality", 0)
                label = f"📊 Quality: {quality:.2f}"
                title_text = f"Analysis: Quality {quality:.2f}, Sharpness {properties.get('sharpness', 0):.2f}"
            elif "Address" in labels:
                addr = properties.get("full_address", "Unknown")
                label = f"🏠 {addr[:15]}..." if len(addr) > 15 else f"🏠 {addr}"
                title_text = f"Address: {addr}"
            elif "Dance" in labels:
                dance_type = properties.get("label", "Unknown")
                label = f"💃 {dance_type}"
                title_text = f"Dance: {dance_type}"
            else:
                label = f"{labels[0]}_{node_id}" if labels else f"Node_{node_id}"
                title_text = f"{labels[0] if labels else 'Node'}: {label}"
            
            # Add node
            node_size = 20
            if highlight_nodes and node_id in highlight_nodes:
                node_size = 35

            net.add_node(
                node_id,
                label=label,
                title=title_text,
                color=color,
                size=node_size
            )
        
        # Add edges with more descriptive labels
        edge_colors = {
            "CONTAINS": "#e74c3c",
            "IDENTIFIED_AS": "#2ecc71",
            "AT_LOCATION": "#3498db",
            "TAKEN_AT": "#f39c12",
            "HAS_CAMERA": "#9b59b6",
            "HAS_TECH": "#1abc9c",
            "HAS_ANALYSIS": "#34495e",
            "RESOLVED_AS": "#e67e22",
            "PERFORMS": "#8e44ad"
        }
        
        edge_labels = {
            "CONTAINS": "contains",
            "IDENTIFIED_AS": "is",
            "AT_LOCATION": "at",
            "TAKEN_AT": "taken",
            "HAS_CAMERA": "with",
            "HAS_TECH": "settings",
            "HAS_ANALYSIS": "analyzed",
            "RESOLVED_AS": "resolves to",
            "PERFORMS": "performs"
        }
        
        for rel in graph_data.get("relationships", []):
            rel_type = rel["type"]
            color = edge_colors.get(rel_type, "#95a5a6")
            G.add_edge(rel["source"], rel["target"])
            
            # Create edge title and label
            properties = rel.get("properties", {})
            edge_label = edge_labels.get(rel_type, rel_type.lower())
            title_text = f"{rel_type}"
            if properties:
                title_text += f": {properties}"
            
            net.add_edge(
                rel["source"],
                rel["target"],
                label=(edge_label if show_edge_labels else ""),
                title=title_text,
                color=color,
                width=3
            )

        # Degree filter and centrality scaling
        try:
            if min_degree > 0:
                low_nodes = [n for n, d in dict(G.degree()).items() if d < min_degree]
                for n in low_nodes:
                    if n in net.nodes_dict:
                        net.nodes = [m for m in net.nodes if m["id"] != n]
                # Also remove connected edges
                net.edges = [e for e in net.edges if e["from"] not in low_nodes and e["to"] not in low_nodes]

            if scale_by_centrality and len(G) > 0:
                cent = nx.degree_centrality(G)
                # Scale to 10..40
                vals = list(cent.values()) or [0]
                cmin, cmax = (min(vals), max(vals)) if vals else (0, 0)
                for n in net.nodes:
                    v = cent.get(n["id"], 0.0)
                    size = 10 + (30 * ((v - cmin) / (cmax - cmin))) if cmax > cmin else 20
                    n["size"] = max(10, min(40, size))
        except Exception:
            pass
        
        # Generate HTML
        html = net.generate_html()
        
        # Extract just the body content for embedding
        start = html.find('<body>') + 6
        end = html.find('</body>')
        body_content = html[start:end]
        
        return body_content
    
    def create_person_network(self, person_data: Dict[str, Any], 
                            person_name: str) -> str:
        """Create a focused network for a specific person."""
        # Create a simplified graph structure
        nodes = []
        relationships = []
        
        # Add person node
        nodes.append({
            "id": "person",
            "labels": ["Person"],
            "properties": {"name": person_name}
        })
        
        # Add images
        for i, img in enumerate(person_data.get("images", [])):
            img_id = f"img_{i}"
            nodes.append({
                "id": img_id,
                "labels": ["Image"],
                "properties": img
            })
            relationships.append({
                "source": img_id,
                "target": "person",
                "type": "CONTAINS_PERSON",
                "properties": {}
            })
        
        # Add locations
        for i, loc in enumerate(person_data.get("locations", [])):
            loc_id = f"loc_{i}"
            nodes.append({
                "id": loc_id,
                "labels": ["Location"],
                "properties": loc
            })
            # Connect to images at this location
            for j, img in enumerate(person_data.get("images", [])):
                if (loc.get("lat") and loc.get("lon") and 
                    img.get("location", {}).get("lat") == loc.get("lat")):
                    relationships.append({
                        "source": f"img_{j}",
                        "target": loc_id,
                        "type": "AT_LOCATION",
                        "properties": {}
                    })
        
        graph_data = {"nodes": nodes, "relationships": relationships}
        return self.create_interactive_network(graph_data, f"Network: {person_name}")
    
    def create_location_heatmap_data(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create data for location heatmap visualization."""
        locations = location_data.get("most_photographed", [])
        
        heatmap_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for loc in locations:
            if loc.get("lat") and loc.get("lon"):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [loc["lon"], loc["lat"]]
                    },
                    "properties": {
                        "photo_count": loc.get("photo_count", 0),
                        "intensity": min(loc.get("photo_count", 0) / 10, 1.0)  # Normalize
                    }
                }
                heatmap_data["features"].append(feature)
        
        return heatmap_data
    
    def create_statistics_summary(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of graph statistics."""
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        # Count nodes by type
        node_counts = {}
        for node in nodes:
            for label in node["labels"]:
                node_counts[label] = node_counts.get(label, 0) + 1
        
        # Count relationships by type
        rel_counts = {}
        for rel in relationships:
            rel_type = rel["type"]
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
        # Calculate density
        n_nodes = len(nodes)
        n_edges = len(relationships)
        density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        
        return {
            "total_nodes": n_nodes,
            "total_relationships": n_edges,
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "density": density,
            "avg_degree": (2 * n_edges) / n_nodes if n_nodes > 0 else 0
        }
