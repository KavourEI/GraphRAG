# GraphRAG

A Retrieval-Augmented Generation (RAG) chatbot using Ollama and GraphDB (Ontotext).

This project implements a knowledge graph-based question answering system that:
- Integrates an LLM and embedding model via Ollama
- Retrieves structured data from a GraphDB knowledge graph
- Generates natural language responses based on retrieved knowledge

## Features

- **GraphDB Integration**: Connect to GraphDB (Ontotext) and execute SPARQL queries
- **Query Analysis**: Automatically identify relevant graphs and predicates from natural language
- **Count Queries**: Ask about available wood types and get graph-based counts
- **Semantic Search**: Use embeddings for relevance-based filtering of retrieved data
- **LLM Response Generation**: Generate natural language answers using Ollama
- **Hallucination Prevention**: Strict context adherence prevents making up information
  - System prompts enforce ONLY using provided context
  - Lower temperature (0.3) reduces creative hallucination
  - Clear "I don't know" responses when data is missing

## Installation

```bash
# Clone the repository
git clone https://github.com/KavourEI/GraphRAG.git
cd GraphRAG

# Install dependencies
pip install -r requirements.txt
```

## Prerequisites

1. **GraphDB (Ontotext)** running at `http://localhost:7200`
   - Create a repository named `Final_W2W_Onto`
   - Load RDF data with wood type information

2. **Ollama** running at `http://localhost:11434`
   - Install Ollama: https://ollama.ai/
   - Pull required models:
     ```bash
     ollama pull llama3.2
     ollama pull nomic-embed-text
     ```

## Data Organization

Each wood type is represented by its own named graph:
- Graph URI format: `http://w2w_onto.com/init/{wood_type}`
- Example: `http://w2w_onto.com/init/oak`

Each graph contains RDF triples describing properties:
```turtle
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://w2w_onto.com/init/oak#material_properties>
    rdfs:hasValue "High density, excellent durability, resistant to moisture" .
```

## Quick Start

```python
from graphrag import get_answer_from_graphdb

# Ask a question about wood properties
answer = get_answer_from_graphdb("What are the material properties of oak tree?")
print(answer)
```

## Usage Examples

### Simple Usage

```python
from graphrag import get_answer_from_graphdb

# Default configuration
answer = get_answer_from_graphdb(
    query="What are the material properties of oak tree?",
)
print(answer)
```

### Querying Available Wood Types

The system now supports general queries about available wood types/graphs:

```python
from graphrag import get_answer_from_graphdb

# Ask about available wood types
answer = get_answer_from_graphdb(
    query="How many wood types do you have information about?",
)
print(answer)
# Example output (actual results depend on your database contents):
# "I have information about 10 wood types (graphs) in the knowledge graph: ash, birch, cedar, cherry, maple, mahogany, oak, pine, teak, walnut."
```

### Custom Configuration

```python
from graphrag import GraphRAGPipeline, GraphDBConnector, QueryAnalyzer, OllamaClient

# Create custom components
graphdb = GraphDBConnector(
    endpoint_url="http://localhost:7200",
    repository="Final_W2W_Onto",
)

analyzer = QueryAnalyzer()
analyzer.add_wood_type("redwood", "http://w2w_onto.com/init/redwood")

ollama = OllamaClient(
    base_url="http://localhost:11434",
    llm_model="llama3.2",
)

# Create pipeline
pipeline = GraphRAGPipeline(
    graphdb_connector=graphdb,
    query_analyzer=analyzer,
    ollama_client=ollama,
)

# Query
answer = pipeline.get_answer("Tell me about maple wood characteristics")
print(answer)
```

### Query Analysis Only

```python
from graphrag import QueryAnalyzer

analyzer = QueryAnalyzer()

result = analyzer.analyze_query("What are the material properties of oak tree?")
print(f"Wood Type: {result['wood_type']}")        # oak
print(f"Graph URI: {result['graph_uri']}")        # http://w2w_onto.com/init/oak
print(f"Property Type: {result['property_type']}") # material_properties
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Analyzer  │ → Identify wood type and property type
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ GraphDB         │ → Retrieve relevant RDF triples
│ Connector       │   via SPARQL queries
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Ollama Client   │ → Filter by relevance (embeddings)
│ (Embeddings)    │   
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Context         │ → Compose context from triples
│ Composition     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Ollama Client   │ → Generate natural language response
│ (LLM)           │
└─────────────────┘
    │
    ▼
Natural Language Response
```

## API Reference

### `get_answer_from_graphdb(query, **kwargs) -> str`

Main function to get an answer from GraphDB using RAG.

**Parameters:**
- `query` (str): The user's natural language question
- `graphdb_endpoint` (str): GraphDB endpoint URL (default: "http://localhost:7200")
- `graphdb_repository` (str): GraphDB repository name (default: "Final_W2W_Onto")
- `ollama_url` (str): Ollama API URL (default: "http://localhost:11434")
- `embedding_model` (str): Ollama model for embeddings (default: "nomic-embed-text")
- `llm_model` (str): Ollama model for response generation (default: "llama3.2")
- `use_embeddings` (bool): Whether to use embeddings for semantic search (default: True)

**Returns:** A natural language response based on knowledge graph data.

### `GraphRAGPipeline`

The main pipeline class that orchestrates the RAG process.

### `GraphDBConnector`

Handles connection and SPARQL queries to GraphDB.

### `QueryAnalyzer`

Analyzes natural language queries to extract relevant information.

### `OllamaClient`

Client for Ollama API (embeddings and LLM).

## Running Examples

```bash
# Run query analysis example (no external services required)
python example.py

# Run all examples (requires GraphDB and Ollama)
python example.py --all

# Interactive mode
python example.py --interactive
```

## Supported Wood Types

The default configuration supports these wood types:
- oak, pine, maple, birch, walnut
- cherry, mahogany, teak, cedar, ash

Add custom wood types using `QueryAnalyzer.add_wood_type()`.

## License

MIT License
