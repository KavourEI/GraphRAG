"""
GraphRAG: A Retrieval-Augmented Generation chatbot using Ollama and GraphDB.

This package provides a modular implementation for:
- Connecting to GraphDB (Ontotext) and executing SPARQL queries
- Analyzing user queries to identify relevant graphs and predicates
- Integrating with Ollama for embeddings and LLM response generation
- Building a complete RAG pipeline for knowledge graph-based Q&A
"""

from graphrag.rag_pipeline import get_answer_from_graphdb, GraphRAGPipeline
from graphrag.graphdb_connector import GraphDBConnector
from graphrag.query_analyzer import QueryAnalyzer
from graphrag.ollama_client import OllamaClient

__version__ = "0.1.0"
__all__ = [
    "get_answer_from_graphdb",
    "GraphRAGPipeline",
    "GraphDBConnector",
    "QueryAnalyzer",
    "OllamaClient",
]
