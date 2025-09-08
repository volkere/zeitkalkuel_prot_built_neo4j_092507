"""
Enhanced graph visualization utilities for Neo4j data.
"""
from typing import Dict, Any, List, Optional, Union
import networkx as nx
from pyvis.network import Network
import json
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re


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
        try:
            html = net.generate_html()
            
            # Extract just the body content for embedding
            start = html.find('<body>') + 6
            end = html.find('</body>')
            if start > 5 and end > start:
                body_content = html[start:end]
                return body_content
            else:
                # Fallback: return full HTML if body extraction fails
                return html
        except Exception as e:
            # Return error message as HTML
            return f'<div style="color: red; padding: 20px;">Fehler bei der Visualisierung: {str(e)}</div>'
    
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
    
    def create_static_network(self, graph_data: Dict[str, Any], title: str = "Neo4j Graph") -> str:
        """Create a static network visualization using matplotlib."""
        try:
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in graph_data.get("nodes", []):
                node_id = node["id"]
                labels = node["labels"]
                G.add_node(node_id, labels=labels, properties=node["properties"])
            
            # Add edges
            for rel in graph_data.get("relationships", []):
                G.add_edge(rel["source"], rel["target"], type=rel["type"])
            
            if len(G.nodes()) == 0:
                return "Keine Knoten zum Visualisieren gefunden."
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Layout
            if len(G.nodes()) > 100:
                pos = nx.spring_layout(G, k=1, iterations=50)
            else:
                pos = nx.spring_layout(G, k=2, iterations=100)
            
            # Color mapping
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
            
            # Draw nodes
            node_colors_list = []
            for node in G.nodes():
                labels = G.nodes[node].get("labels", [])
                color = "#95a5a6"  # default gray
                for label in labels:
                    if label in node_colors:
                        color = node_colors[label]
                        break
                node_colors_list.append(color)
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=300, alpha=0.7)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
            
            # Add labels for small graphs
            if len(G.nodes()) <= 50:
                labels = {}
                for node in G.nodes():
                    props = G.nodes[node].get("properties", {})
                    node_labels = G.nodes[node].get("labels", [])
                    if "Person" in node_labels and props.get("name"):
                        labels[node] = props["name"][:10]
                    elif "Image" in node_labels and props.get("name"):
                        labels[node] = props["name"][:10]
                    else:
                        labels[node] = str(node)
                nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title(f"{title} ({len(G.nodes())} Knoten, {len(G.edges())} Kanten)")
            plt.axis('off')
            
            # Create legend
            legend_elements = []
            for label, color in node_colors.items():
                legend_elements.append(mpatches.Patch(color=color, label=label))
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # Convert to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f'<img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 1000px;">'
            
        except Exception as e:
            return f'<div style="color: red; padding: 20px;">Fehler bei der statischen Visualisierung: {str(e)}</div>'
    
    def parse_cypher_query(self, query: str) -> Dict[str, Any]:
        """Parse a Cypher query and extract nodes, relationships, and patterns."""
        try:
            # Clean up query
            query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)  # Remove comments
            query = re.sub(r'\s+', ' ', query.strip())  # Normalize whitespace
            
            # Extract MATCH patterns
            match_patterns = re.findall(r'MATCH\s+([^WHERE|RETURN|WITH]+)', query, re.IGNORECASE)
            
            nodes = []
            relationships = []
            node_counter = 0
            rel_counter = 0
            
            for pattern in match_patterns:
                # Parse node patterns like (n:Label) or (n:Label {prop: value})
                node_matches = re.findall(r'\(([^)]+)\)', pattern)
                
                for node_match in node_matches:
                    # Extract variable name and labels
                    var_match = re.match(r'(\w+)(?::([^\{]+))?(?:\s*\{([^}]+)\})?', node_match.strip())
                    if var_match:
                        var_name = var_match.group(1)
                        labels = var_match.group(2).split(':') if var_match.group(2) else ['Node']
                        props = var_match.group(3) if var_match.group(3) else ''
                        
                        nodes.append({
                            "id": var_name,
                            "labels": [label.strip() for label in labels if label.strip()],
                            "properties": {"variable": var_name, "props": props}
                        })
                
                # Parse relationship patterns like -[r:TYPE]-> or -[r:TYPE {prop: value}]->
                rel_matches = re.findall(r'-\[([^\]]+)\]->?', pattern)
                
                for rel_match in rel_matches:
                    # Extract relationship variable and type
                    rel_var_match = re.match(r'(\w+)(?::([^\{]+))?(?:\s*\{([^}]+)\})?', rel_match.strip())
                    if rel_var_match:
                        rel_var = rel_var_match.group(1)
                        rel_type = rel_var_match.group(2) if rel_var_match.group(2) else 'RELATED_TO'
                        rel_props = rel_var_match.group(3) if rel_var_match.group(3) else ''
                        
                        relationships.append({
                            "id": f"rel_{rel_counter}",
                            "type": rel_type.strip(),
                            "properties": {"variable": rel_var, "props": rel_props}
                        })
                        rel_counter += 1
            
            # Extract RETURN clause
            return_clause = re.search(r'RETURN\s+([^ORDER|LIMIT]+)', query, re.IGNORECASE)
            return_vars = []
            if return_clause:
                return_vars = [var.strip() for var in return_clause.group(1).split(',')]
            
            # Extract WHERE conditions
            where_clause = re.search(r'WHERE\s+([^RETURN|WITH]+)', query, re.IGNORECASE)
            where_conditions = []
            if where_clause:
                where_conditions = [where_clause.group(1).strip()]
            
            return {
                "nodes": nodes,
                "relationships": relationships,
                "return_variables": return_vars,
                "where_conditions": where_conditions,
                "original_query": query
            }
            
        except Exception as e:
            return {"error": f"Fehler beim Parsen der Cypher-Query: {str(e)}"}
    
    def visualize_cypher_query(self, query: str) -> str:
        """Create a visual representation of a Cypher query structure."""
        try:
            parsed = self.parse_cypher_query(query)
            if "error" in parsed:
                return f'<div style="color: red; padding: 20px;">{parsed["error"]}</div>'
            
            # Create NetworkX graph for query visualization
            G = nx.DiGraph()
            
            # Add nodes
            for node in parsed["nodes"]:
                G.add_node(node["id"], labels=node["labels"], properties=node["properties"])
            
            # Add relationships (simplified - just connect nodes in order)
            nodes = parsed["nodes"]
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i]["id"], nodes[i+1]["id"], type="RELATED")
            
            if len(G.nodes()) == 0:
                return '<div style="color: orange; padding: 20px;">Keine Knoten in der Query gefunden.</div>'
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Layout
            pos = nx.spring_layout(G, k=3, iterations=100)
            
            # Color mapping for query elements
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
                "Dance": "#00b894",
                "Node": "#95a5a6"
            }
            
            # Draw nodes
            node_colors_list = []
            for node in G.nodes():
                labels = G.nodes[node].get("labels", ["Node"])
                color = "#95a5a6"  # default
                for label in labels:
                    if label in node_colors:
                        color = node_colors[label]
                        break
                node_colors_list.append(color)
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=1000, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color='#2c3e50', arrows=True, arrowsize=30, width=3, alpha=0.7)
            
            # Add labels
            labels = {}
            for node in G.nodes():
                node_labels = G.nodes[node].get("labels", ["Node"])
                props = G.nodes[node].get("properties", {})
                var_name = props.get("variable", node)
                label_text = f"{var_name}\n{':'.join(node_labels)}"
                if props.get("props"):
                    label_text += f"\n{{{props['props'][:20]}...}}"
                labels[node] = label_text
            
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
            
            # Add title with query info
            title = f"Cypher Query Structure\n{len(parsed['nodes'])} Knoten, {len(parsed['relationships'])} Beziehungen"
            if parsed["return_variables"]:
                title += f"\nRETURN: {', '.join(parsed['return_variables'])}"
            if parsed["where_conditions"]:
                title += f"\nWHERE: {parsed['where_conditions'][0][:50]}..."
            
            plt.title(title, fontsize=12, pad=20)
            plt.axis('off')
            
            # Create legend
            legend_elements = []
            for label, color in node_colors.items():
                if any(label in node.get("labels", []) for node in parsed["nodes"]):
                    legend_elements.append(mpatches.Patch(color=color, label=label))
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # Add query text box
            query_text = parsed["original_query"][:200] + "..." if len(parsed["original_query"]) > 200 else parsed["original_query"]
            plt.figtext(0.02, 0.02, f"Query: {query_text}", fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            
            # Convert to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f'<img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 1200px;">'
            
        except Exception as e:
            return f'<div style="color: red; padding: 20px;">Fehler bei der Query-Visualisierung: {str(e)}</div>'
