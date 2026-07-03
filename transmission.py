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
    NodeSchema("School", ["schoolName"], "Represents the school name"),
    NodeSchema("Address", ["cityName", "streetName", "streetNumber", "postalCode"],
               "Represents school's postal address"),
]

relationship_schemas = [
    RelationshipSchema("School", "isPlacedInto", "Address", [])
]

additional_instructions = """
- all names must be extracted as uppercase"""

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)

llm = ChatOllama(
    base_url=url,
    model="qwen3.5:35b",
    temperature=0,
    keep_alive="10m",
)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=node_schemas,
    allowed_relationships=relationship_schemas,
    additional_instructions=additional_instructions
)

filepath = Path("pdfs") / "1.ΔΙΑΒΙΒΑΣΤΙΚΟ.pdf"
output_file = Path("pdfs") / "graph_output.txt"

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