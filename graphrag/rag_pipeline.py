"""
RAG Pipeline Module.

Implements the complete Retrieval-Augmented Generation pipeline that:
1. Accepts a natural language question
2. Identifies the relevant graph for the query
3. Queries GraphDB for relevant triples
4. Retrieves triples with predicates containing "hasValue"
5. Passes retrieved data as context to Ollama for answer generation
6. Returns a natural-language response
"""

import logging
from typing import Optional

from graphrag.graphdb_connector import GraphDBConnector
from graphrag.query_analyzer import QueryAnalyzer
from graphrag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Pattern used to identify wood type graphs in the knowledge base
# Each wood type has its own named graph with URIs following the pattern:
# http://w2w_onto.com/init/{wood_type}
WOOD_GRAPH_PATTERN = "/init/"


class GraphRAGPipeline:
    """
    A complete RAG pipeline integrating GraphDB and Ollama.
    
    This class orchestrates the entire process of:
    - Analyzing user queries
    - Retrieving relevant knowledge from GraphDB
    - Using embeddings for semantic search
    - Generating natural language responses via Ollama
    """

    def __init__(
        self,
        graphdb_connector: Optional[GraphDBConnector] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        ollama_client: Optional[OllamaClient] = None,
        graphdb_endpoint: str = "http://localhost:7200",
        graphdb_repository: str = "Final_W2W_Onto",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        use_embeddings: bool = True,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            graphdb_connector: Optional pre-configured GraphDB connector.
            query_analyzer: Optional pre-configured query analyzer.
            ollama_client: Optional pre-configured Ollama client.
            graphdb_endpoint: GraphDB endpoint URL.
            graphdb_repository: GraphDB repository name.
            ollama_url: Ollama API URL.
            embedding_model: Ollama model for embeddings.
            llm_model: Ollama model for response generation.
            use_embeddings: Whether to use embeddings for semantic search.
        """
        self.graphdb = graphdb_connector or GraphDBConnector(
            endpoint_url=graphdb_endpoint,
            repository=graphdb_repository,
        )
        self.analyzer = query_analyzer or QueryAnalyzer()
        self.ollama = ollama_client or OllamaClient(
            base_url=ollama_url,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )
        self.use_embeddings = use_embeddings

    def get_answer(self, query: str) -> str:
        """
        Process a user query and return a natural language answer.

        This is the main entry point for the RAG pipeline.

        Args:
            query: The user's natural language question.

        Returns:
            A natural language response based on knowledge graph data.
        """
        logger.info(f"Processing query: {query}")

        # Step 1: Analyze the query to identify graph and predicates
        analysis = self.analyzer.analyze_query(query)
        
        # Handle general count queries about available wood types/graphs
        if analysis.get("is_general_count_query"):
            return self._generate_count_response(query)
        
        if not analysis["graph_uri"]:
            return self._generate_no_graph_response(query)

        # Step 2: Retrieve relevant triples from GraphDB
        context_triples = self._retrieve_triples(analysis)
        
        if not context_triples:
            return self._generate_no_data_response(query, analysis)

        # Step 3: Optionally filter using embeddings for relevance
        if self.use_embeddings:
            context_triples = self._filter_by_relevance(query, context_triples)

        # Step 4: Compose context from retrieved triples
        context = self._compose_context(context_triples, analysis)

        # Step 5: Generate response using Ollama
        response = self._generate_response(query, context, analysis)

        return response

    def _retrieve_triples(self, analysis: dict) -> list[dict]:
        """
        Retrieve relevant triples from GraphDB based on query analysis.

        Args:
            analysis: The query analysis result.

        Returns:
            A list of triple dictionaries with keys:
                - 's': Subject URI
                - 'p': Predicate URI
                - 'o': Object value (literal or URI)
        """
        graph_uri = analysis["graph_uri"]
        triples = []

        try:
            # If a property type is identified, get ALL triples about that subject
            if analysis.get("property_type"):
                # Get all triples where this property is the subject
                subject_triples = self.graphdb.get_triples_by_subject(
                    graph_uri=graph_uri,
                    subject_contains=analysis["property_type"],
                )
                triples.extend(subject_triples)
                logger.info(f"Retrieved {len(subject_triples)} triples for subject containing '{analysis['property_type']}'")

            # Also get triples with hasValue predicate (as specified in requirements)
            value_triples = self.graphdb.get_values_by_predicate(
                graph_uri=graph_uri,
                predicate_contains="hasValue",
                subject_filter=analysis.get("property_type"),
            )
            triples.extend(value_triples)

            # Also search for property-related triples
            if analysis.get("property_type"):
                property_triples = self.graphdb.get_triples_from_graph(
                    graph_uri=graph_uri,
                    predicate_filter=analysis["property_type"],
                )
                triples.extend(property_triples)

            # Search for additional relevant triples using keywords
            for term in analysis.get("search_terms", []):
                search_triples = self.graphdb.search_triples(
                    graph_uri=graph_uri,
                    search_term=term,
                    limit=20,
                )
                triples.extend(search_triples)

            # Deduplicate triples
            seen = set()
            unique_triples = []
            for triple in triples:
                key = (triple.get("s"), triple.get("p"), triple.get("o"))
                if key not in seen:
                    seen.add(key)
                    unique_triples.append(triple)

            logger.info(f"Retrieved {len(unique_triples)} unique triples from {graph_uri}")
            return unique_triples

        except Exception as e:
            logger.error(f"Failed to retrieve triples: {e}")
            return []

    def _filter_by_relevance(
        self,
        query: str,
        triples: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Filter triples by semantic relevance using embeddings.

        Args:
            query: The user query.
            triples: List of triples to filter.
            top_k: Number of most relevant triples to keep.

        Returns:
            Filtered list of most relevant triples.
        """
        if not triples or len(triples) <= top_k:
            return triples

        try:
            # Generate query embedding
            query_embedding = self.ollama.generate_embedding(query)

            # Generate embeddings for triple representations
            triple_texts = [self._triple_to_text(t) for t in triples]
            triple_embeddings = self.ollama.generate_embeddings_batch(triple_texts)

            # Find most similar triples
            similar_indices = self.ollama.find_most_similar(
                query_embedding,
                triple_embeddings,
                top_k=top_k,
            )

            # Return filtered triples
            return [triples[i] for i, _ in similar_indices]

        except Exception as e:
            logger.warning(f"Embedding-based filtering failed: {e}. Returning all triples.")
            return triples[:top_k]

    def _triple_to_text(self, triple: dict) -> str:
        """
        Convert a triple to a human-readable text representation.

        Args:
            triple: A dictionary with 's', 'p', 'o' keys.

        Returns:
            A text representation of the triple.
        """
        subject = self._extract_local_name(triple.get("s", ""))
        predicate = self._extract_local_name(triple.get("p", ""))
        obj = triple.get("o", "")
        
        # Clean up the predicate for readability
        predicate = predicate.replace("_", " ").replace("-", " ")
        
        return f"{subject} {predicate}: {obj}"

    def _extract_local_name(self, uri: str) -> str:
        """
        Extract the local name from a URI.

        Args:
            uri: A URI string.

        Returns:
            The local name (part after the last / or #).
        """
        if "#" in uri:
            return uri.split("#")[-1]
        if "/" in uri:
            return uri.split("/")[-1]
        return uri

    def _compose_context(self, triples: list[dict], analysis: dict) -> str:
        """
        Compose a context string from retrieved triples.

        Args:
            triples: List of triples from the knowledge graph.
            analysis: The query analysis result.

        Returns:
            A formatted context string for the LLM.
        """
        wood_type = analysis.get("wood_type", "unknown")
        
        context_parts = [
            f"Information about {wood_type} wood from the knowledge graph:",
            "",
        ]

        for triple in triples:
            text = self._triple_to_text(triple)
            context_parts.append(f"- {text}")

        return "\n".join(context_parts)

    def _generate_response(
        self,
        query: str,
        context: str,
        analysis: dict,
    ) -> str:
        """
        Generate a natural language response using Ollama.

        Args:
            query: The user's original query.
            context: The context composed from retrieved triples.
            analysis: The query analysis result.

        Returns:
            A natural language response.
        """
        system_prompt = """You are a helpful assistant specializing in wood and material science. 
You answer questions STRICTLY based on the provided knowledge graph data.

CRITICAL INSTRUCTIONS:
- ONLY use information from the provided context to answer questions
- If the context doesn't contain the information needed to answer the question, you MUST say "I don't have that information in my knowledge base"
- DO NOT make up, infer, or assume any information that is not explicitly stated in the context
- DO NOT use general knowledge about wood or materials that is not in the context
- Be concise, accurate, and helpful with the information that IS available
- If you can only partially answer the question based on the context, clearly state what you can answer and what you cannot"""

        prompt = f"""Context from Knowledge Graph:
{context}

User Question: {query}

Please provide a helpful answer based ONLY on the context above. If the context does not contain the information needed to answer the question, clearly state that you don't have that information."""

        try:
            response = self.ollama.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature to reduce hallucination
            )
            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while generating the response: {e}"

    def _generate_count_response(self, query: str) -> str:
        """
        Generate a response for queries about available wood types/graphs.

        Args:
            query: The user's query.

        Returns:
            A response with the count of available graphs/wood types.
        """
        try:
            # Get all named graphs from the database
            all_graphs = self.graphdb.list_named_graphs()
            
            # Filter graphs that match the wood type pattern
            wood_graphs = [g for g in all_graphs if WOOD_GRAPH_PATTERN in g]
            
            # Extract wood type names from graph URIs
            wood_types = []
            for graph in wood_graphs:
                # Extract the wood type from URIs like "http://w2w_onto.com/init/oak"
                wood_type = graph.split(WOOD_GRAPH_PATTERN)[-1]
                wood_types.append(wood_type)
            
            count = len(wood_types)
            
            if count == 0:
                return "I currently don't have information about any wood types in the knowledge graph."
            elif count == 1:
                return f"I have information about 1 wood type (graph) in the knowledge graph: {wood_types[0]}."
            else:
                wood_list = ", ".join(sorted(wood_types))
                return f"I have information about {count} wood types (graphs) in the knowledge graph: {wood_list}."
                
        except Exception as e:
            logger.error(f"Failed to retrieve graph count: {e}")
            # Fallback to known wood types from analyzer
            known_types = sorted(self.analyzer.wood_type_graphs.keys())
            count = len(known_types)
            wood_list = ", ".join(known_types)
            return f"I have information about {count} wood types (graphs): {wood_list}. (Note: This is based on configured types; actual database may vary.)"

    def _generate_no_graph_response(self, query: str) -> str:
        """
        Generate a response when no relevant graph is found.

        Args:
            query: The user's query.

        Returns:
            A helpful response explaining the limitation.
        """
        available_types = ", ".join(self.analyzer.wood_type_graphs.keys())
        return (
            f"I couldn't identify a specific wood type in your question. "
            f"Please mention a wood type in your query. "
            f"Available wood types include: {available_types}."
        )

    def _generate_no_data_response(self, query: str, analysis: dict) -> str:
        """
        Generate a response when no data is found in the graph.

        Args:
            query: The user's query.
            analysis: The query analysis result.

        Returns:
            A helpful response explaining that no data was found.
        """
        wood_type = analysis.get("wood_type", "the specified wood")
        return (
            f"I don't have information to answer your question about {wood_type}. "
            f"While I found the graph for {wood_type}, it doesn't contain "
            f"data relevant to your specific question."
        )


def get_answer_from_graphdb(
    query: str,
    graphdb_endpoint: str = "http://localhost:7200",
    graphdb_repository: str = "Final_W2W_Onto",
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "llama3.2",
    use_embeddings: bool = True,
) -> str:
    """
    Main function to get an answer from GraphDB using RAG.

    This is the primary entry point for the GraphRAG system.

    Args:
        query: The user's natural language question.
        graphdb_endpoint: GraphDB endpoint URL.
        graphdb_repository: GraphDB repository name.
        ollama_url: Ollama API URL.
        embedding_model: Ollama model for embeddings.
        llm_model: Ollama model for response generation.
        use_embeddings: Whether to use embeddings for semantic search.

    Returns:
        A natural language response based on knowledge graph data.

    Example:
        >>> answer = get_answer_from_graphdb("What are the material properties of oak tree?")
        >>> print(answer)
    """
    pipeline = GraphRAGPipeline(
        graphdb_endpoint=graphdb_endpoint,
        graphdb_repository=graphdb_repository,
        ollama_url=ollama_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        use_embeddings=use_embeddings,
    )
    return pipeline.get_answer(query)
