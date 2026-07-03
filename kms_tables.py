from LLMGraphTransformer import LLMGraphTransformer
from LLMGraphTransformer.schema import NodeSchema, RelationshipSchema
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from extract_from_pdf import extraction, extraction_person
import os
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

url = os.getenv('RUNPOD_URL')

node_schemas = [
    NodeSchema("Person", ["lastName", "firstName", "fatherName", "taxIdentificationNumber"], "Represents an individual person"),
    NodeSchema("Travel", ["routeNumber", "travelDate", "travelDay", "fromCity1", "toCity1", "fromCity2", "toCity2", "distanceKM"], "Represents the travel distance between two cities"),
]

relationship_schemas = [
    RelationshipSchema("Person", "Undertakes", "Travel", ["startDate", "endDate", "travelDay"])
]

additional_instructions = """
- all names must be extracted as uppercase
- No Greeklish allowed, only Greek

IMPORTANT:
- Every row of the mileage table is a separate Travel node.
- Never merge rows even if route, cities or distance are identical.
- Create one Travel node per route number.
- Travel nodes must contain routeNumber and travelDate.
- Create one relationship for every table row.
- Always add fromCity and toCity to the Travel node.
- id for Person 'lastName firstName'
"""

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)

llm = ChatOllama(
    base_url=url,
    model="gpt-oss:120b",
    temperature=0,
    keep_alive="10m",
)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=node_schemas,
    allowed_relationships=relationship_schemas,
    additional_instructions=additional_instructions
)

filepath = Path("pdfs") / "3.Παρουσιολόγιο – Πίνακας Χιλιομέτρων.pdf"
output_file = Path("pdfs") / "graph_output3.txt"

if not filepath.exists():
    print(f"File not found: {filepath.resolve()}")
elif not os.access(filepath, os.R_OK):
    print(f"No read permission: {filepath.resolve()}")
else:
    try:
        # rendered = converter(str(filepath))
        # text, _, images = text_from_rendered(rendered)
        text = extraction_person(filepath, url)
        document = Document(page_content=text)
        graph_document = llm_transformer.convert_to_graph_document(document)

        print(f"Nodes: {graph_document.nodes}")
        print(f"Relationships: {graph_document.relationships}")
        print(f"Graph Document: {graph_document}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(graph_document))

        print(f"✅ Αποθηκεύτηκε: {output_file}")

    except Exception as e:
        print(f"❌ Σφάλμα στο {filepath}: {e}")