#!/usr/bin/env python3
"""
Example script demonstrating the GraphRAG pipeline.

This script shows how to use the GraphRAG system to query a knowledge graph
about wood types and get natural language responses.

Prerequisites:
1. GraphDB (Ontotext) running at http://localhost:7200
2. Ollama running at http://localhost:11434
3. A repository named "wood_types" with wood data graphs

Usage:
    python example.py

Or in Python:
    from graphrag import get_answer_from_graphdb
    answer = get_answer_from_graphdb("What are the material properties of oak tree?")
"""

import logging
import sys

from graphrag import (
    get_answer_from_graphdb,
    GraphRAGPipeline,
    GraphDBConnector,
    QueryAnalyzer,
    OllamaClient,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_simple_usage():
    """
    Demonstrate the simplest way to use GraphRAG.
    """
    print("\n" + "=" * 60)
    print("Example 1: Simple Usage with get_answer_from_graphdb()")
    print("=" * 60)

    query = "What are the material properties of oak tree?"
    print(f"\nQuery: {query}")
    
    try:
        answer = get_answer_from_graphdb(query)
        print(f"\nAnswer:\n{answer}")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure GraphDB and Ollama are running.")


def example_custom_configuration():
    """
    Demonstrate using GraphRAG with custom configuration.
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    # Create custom components
    graphdb = GraphDBConnector(
        endpoint_url="http://localhost:7200",
        repository="wood_types",
        timeout=60,
    )

    analyzer = QueryAnalyzer()
    # Add a custom wood type
    analyzer.add_wood_type("redwood", "http://w2w_onto.com/init/redwood")

    ollama = OllamaClient(
        base_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        llm_model="llama3.2",
    )

    # Create pipeline with custom components
    pipeline = GraphRAGPipeline(
        graphdb_connector=graphdb,
        query_analyzer=analyzer,
        ollama_client=ollama,
        use_embeddings=True,
    )

    query = "Tell me about the characteristics of maple wood"
    print(f"\nQuery: {query}")

    try:
        answer = pipeline.get_answer(query)
        print(f"\nAnswer:\n{answer}")
    except Exception as e:
        print(f"\nError: {e}")


def example_query_analysis():
    """
    Demonstrate query analysis without calling external services.
    """
    print("\n" + "=" * 60)
    print("Example 3: Query Analysis (No External Services Required)")
    print("=" * 60)

    analyzer = QueryAnalyzer()

    queries = [
        "What are the material properties of oak tree?",
        "Tell me about pine wood characteristics",
        "What is the color of maple?",
        "Where does cherry wood come from?",
        "What can I use birch wood for?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        analysis = analyzer.analyze_query(query)
        print(f"  Wood Type: {analysis['wood_type']}")
        print(f"  Graph URI: {analysis['graph_uri']}")
        print(f"  Property Type: {analysis['property_type']}")
        print(f"  Search Terms: {analysis['search_terms']}")


def example_check_services():
    """
    Check if required services are available.
    """
    print("\n" + "=" * 60)
    print("Example 4: Check Service Availability")
    print("=" * 60)

    # Check Ollama
    ollama = OllamaClient()
    if ollama.is_available():
        print("\n✓ Ollama is available")
        models = ollama.list_models()
        if models:
            print(f"  Available models: {', '.join(models[:5])}")
    else:
        print("\n✗ Ollama is not available at http://localhost:11434")
        print("  Please start Ollama: ollama serve")

    # Check GraphDB
    graphdb = GraphDBConnector()
    try:
        graphs = graphdb.list_named_graphs()
        print(f"\n✓ GraphDB is available")
        if graphs:
            print(f"  Available graphs: {len(graphs)}")
            for g in graphs[:5]:
                print(f"    - {g}")
    except Exception as e:
        print(f"\n✗ GraphDB is not available: {e}")
        print("  Please ensure GraphDB is running at http://localhost:7200")


def example_interactive():
    """
    Interactive mode for querying the knowledge graph.
    """
    print("\n" + "=" * 60)
    print("Example 5: Interactive Mode")
    print("=" * 60)
    print("\nEnter your questions about wood types.")
    print("Type 'quit' or 'exit' to stop.\n")

    pipeline = GraphRAGPipeline(use_embeddings=False)  # Faster without embeddings

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            answer = pipeline.get_answer(query)
            print(f"\nAssistant: {answer}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """
    Run all examples.
    """
    print("\n" + "#" * 60)
    print("# GraphRAG Examples")
    print("# Retrieval-Augmented Generation with Ollama and GraphDB")
    print("#" * 60)

    # Example that doesn't require external services
    example_query_analysis()

    # Examples that require external services
    print("\n\nThe following examples require external services (GraphDB and Ollama).")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        example_check_services()
        example_simple_usage()
        example_custom_configuration()
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        example_check_services()
        example_interactive()
    else:
        print("Run with --all to execute all examples")
        print("Run with --interactive for interactive mode")


if __name__ == "__main__":
    main()
