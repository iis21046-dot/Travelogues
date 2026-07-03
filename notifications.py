from LLMGraphTransformer import LLMGraphTransformer
from LLMGraphTransformer.schema import NodeSchema, RelationshipSchema
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from extract_from_pdf import extraction, extraction_person
import os

url = os.getenv('RUNPOD_URL')

node_schemas = [
    NodeSchema("Person", ["last_name", "first_name"], "Represents an individual"),
    NodeSchema("Position", ["Working_hours_per_week", "Branch/specialization", "School_name"],
               "Represents the school/position of the teacher"),
]

relationship_schemas = [
    RelationshipSchema("Person", "isAssignedTo", "Position", ['from_date', "thru_date", "position_assignment_type"])
]

additional_instructions = """
- all names must be extracted as uppercase
- Carefull the document refers to only one teacher, so the other names may be administrative staff
- "branch/Specialization" is the same property for both Person and Position
- "position_type" must have one of following values ["Τοποθέτηση", "Απόσπαση", "Διάθεση"] 
- No Greeklish allowed, only Greek"""

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

directory = os.fsencode('./ΚΟΙΝΟΠΟΙΗΤΗΡΙΑ')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if not filename.endswith(".pdf"):
        continue

    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join('./ΚΟΙΝΟΠΟΙΗΤΗΡΙΑ', f"{base_name}.txt")

    if os.path.exists(output_path):
        print(f"⏭️ Παράλειψη (υπάρχει ήδη): {output_path}")
        continue

    filepath = os.path.join('./ΚΟΙΝΟΠΟΙΗΤΗΡΙΑ', filename)
    print(f"🔄 Επεξεργασία: {filename}")

    try:
        text = extraction_person(filepath, url)
        document = Document(page_content=text)
        graph_document = llm_transformer.convert_to_graph_document(document)

        print(f"Nodes: {graph_document.nodes}")
        print(f"Relationships: {graph_document.relationships}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{graph_document}")

        print(f"✅ Αποθηκεύτηκε: {output_path}")

    except Exception as e:
        print(f"❌ Σφάλμα στο {filename}: {e}")
        continue