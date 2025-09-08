import json
import os
import tempfile
import streamlit as st

from app.graph_persistence import Neo4jPersistence


st.title("Linked Open Data Export")
st.caption("Exportiere Neo4j-Daten als RDF (Turtle/JSON-LD) mit einfachen Ontologie-Mappings")

with st.sidebar:
    st.subheader("Neo4j Verbindung")
    uri = st.text_input("Bolt URI", value=st.session_state.get("neo4j_uri", "bolt://localhost:7687"))
    user = st.text_input("User", value=st.session_state.get("neo4j_user", "neo4j"))
    pwd = st.text_input("Passwort", type="password", value=st.session_state.get("neo4j_pwd", ""))
    base_uri = st.text_input("Base URI für Ressourcen", value="https://example.org/zeitkalkuel/")

col1, col2 = st.columns(2)
fmt = col1.selectbox("Export-Format", ["turtle", "json-ld"], index=0)
limit_info = col2.text_input("Hinweis", value="Es werden alle Knoten/Beziehungen exportiert (bis 100k).")

if st.button("RDF exportieren"):
    try:
        neo = Neo4jPersistence(uri, user, pwd)
        ok = neo.export_rdf(output_file := tempfile.mktemp(suffix=".ttl" if fmt=="turtle" else ".jsonld"), fmt=fmt, base_uri=base_uri)
        neo.close()
        if ok:
            with open(output_file, "rb") as f:
                data = f.read()
            st.success("RDF-Export erfolgreich")
            st.download_button("Download RDF", data=data, file_name=os.path.basename(output_file), mime="text/turtle" if fmt=="turtle" else "application/ld+json")
        else:
            st.error("Export fehlgeschlagen. Bitte Verbindung/Log prüfen.")
    except Exception as e:
        st.error(f"Fehler: {e}")

st.markdown("""
### Mapping-Übersicht
- Image → schema:ImageObject (Titel aus Dateiname)
- Person → foaf:Person (Name)
- Location → schema:Place mit geo:lat/geo:long
- Capture → schema:Event (startDate)
- Address → schema:PostalAddress
- Kanten werden auf schema:* oder zk:* (Projekt-Namespace) gemappt
""")


