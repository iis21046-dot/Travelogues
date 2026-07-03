# Travelogues

# PDF Graph Extraction Pipeline

This project provides a modular Python-based pipeline designed to extract structured information from various PDF documents (such as declarations, notifications, and mileage tables) and convert them into **Knowledge Graph documents** using LangChain and LLMs.

## Overview

The system uses `LLMGraphTransformer` to parse unstructured text extracted from PDFs and map it to specific, predefined node and relationship schemas. It leverages local/remote LLMs via Ollama to handle the extraction logic.

### Key Components

*   **Extraction Module:** Uses custom utilities (`extract_from_pdf`) to convert PDF content into text.
*   **Graph Transformation:** Uses `LLMGraphTransformer` to enforce specific schemas (`NodeSchema`, `RelationshipSchema`) on the extracted data.
*   **Configurable Schemas:** Each script defines its own schema tailored to the specific type of document (e.g., Personnel records, Travel/Mileage reports).
*   **LLM Integration:** Powered by `ChatOllama` (configured to connect to a RunPod URL).

## Project Structure

*   `declaration.py`: Extracts personnel and address data from "Υπεύθυνη Δήλωση" documents.
*   `realDistance.py`: Parses travel distances between cities.
*   `transmission.py`: Extracts school and address information from transmittal documents.
*   `notifications.py`: Processes batch notifications regarding teacher assignments to schools/positions.
*   `kms_tables.py`: Handles complex mileage tables, ensuring every row is captured as a distinct `Travel` node.

## Setup & Requirements

### Prerequisites
*   Python 3.x
*   [Ollama](https://ollama.com/) or a remote LLM endpoint (e.g., RunPod)
*   The following libraries: `langchain`, `langchain-ollama`, `LLMGraphTransformer`, `marker-pdf`

### Environment Variables
You must set the `RUNPOD_URL` environment variable to connect to your LLM instance:
```bash
export RUNPOD_URL="your_runpod_url_here"
```

## Usage

To run a specific extraction task, execute the corresponding script:

```bash
# Example: Process mileage tables
python kms_tables.py

# Example: Process notification documents in batch
python notifications.py
```

### Batch Processing
The `notifications.py` script is designed to iterate through a folder (`./ΚΟΙΝΟΠΟΙΗΤΗΡΙΑ`), automatically skipping files that have already been processed to save tokens and time.

## Customization

### Schemas
To modify the extraction logic, update the `node_schemas` and `relationship_schemas` lists within the respective files. You can also adjust the `additional_instructions` string to provide more specific prompting constraints to the LLM (e.g., forcing uppercase output, restricting field values, or defining specific ID formatting).

### Model Selection
Currently, the scripts are configured to use `qwen3.5:35b` or `gpt-oss:120b`. You can change the `model` parameter in the `ChatOllama` initialization within each file based on your available infrastructure.

## License
[Include your license here, e.g., MIT]
