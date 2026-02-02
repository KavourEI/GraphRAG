"""
Query Analyzer Module.

Analyzes natural language queries to identify relevant graphs and predicates
for retrieval from the knowledge graph.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Default wood types with their corresponding graph URIs
DEFAULT_WOOD_TYPE_GRAPHS = {
    "oak": "http://w2w_onto.com/init/oak",
    "pine": "http://w2w_onto.com/init/pine",
    "maple": "http://w2w_onto.com/init/maple",
    "birch": "http://w2w_onto.com/init/birch",
    "walnut": "http://w2w_onto.com/init/walnut",
    "cherry": "http://w2w_onto.com/init/cherry",
    "mahogany": "http://w2w_onto.com/init/mahogany",
    "teak": "http://w2w_onto.com/init/teak",
    "cedar": "http://w2w_onto.com/init/cedar",
    "ash": "http://w2w_onto.com/init/ash",
}

# Common property patterns to look for in queries
PROPERTY_PATTERNS = {
    "material_properties": [
        "material propert",
        "physical propert",
        "mechanical propert",
        "strength",
        "hardness",
        "density",
        "durability",
    ],
    "color": ["color", "colour", "appearance", "look"],
    "usage": ["use", "application", "purpose", "suitable for"],
    "origin": ["origin", "native", "grow", "found", "region", "country"],
    "characteristics": ["characteristic", "feature", "quality", "trait"],
}


class QueryAnalyzer:
    """
    Analyzes user queries to extract relevant graph URIs and predicates.
    
    This class maps natural language queries to the appropriate knowledge graph
    components for retrieval.
    """

    def __init__(
        self,
        wood_type_graphs: Optional[dict[str, str]] = None,
        property_patterns: Optional[dict[str, list[str]]] = None,
        graph_uri_template: str = "http://w2w_onto.com/init/{wood_type}",
    ):
        """
        Initialize the query analyzer.

        Args:
            wood_type_graphs: Mapping of wood types to their graph URIs.
            property_patterns: Mapping of property names to search patterns.
            graph_uri_template: Template for generating graph URIs from wood types.
        """
        self.wood_type_graphs = wood_type_graphs or DEFAULT_WOOD_TYPE_GRAPHS.copy()
        self.property_patterns = property_patterns or PROPERTY_PATTERNS.copy()
        self.graph_uri_template = graph_uri_template
        
        # Pre-compile regex patterns for wood types for better performance
        self._wood_type_patterns: dict[str, re.Pattern] = {}
        self._compile_wood_type_patterns()

    def _compile_wood_type_patterns(self):
        """Compile regex patterns for all known wood types."""
        for wood_type in self.wood_type_graphs.keys():
            pattern = rf'\b{re.escape(wood_type)}\b'
            self._wood_type_patterns[wood_type] = re.compile(pattern, re.IGNORECASE)

    def extract_wood_type(self, query: str) -> Optional[str]:
        """
        Extract the wood type mentioned in the query.

        Args:
            query: The user's natural language query.

        Returns:
            The detected wood type, or None if not found.
        """
        # Check for known wood types using pre-compiled patterns
        for wood_type, pattern in self._wood_type_patterns.items():
            if pattern.search(query):
                logger.debug(f"Detected wood type: {wood_type}")
                return wood_type
        
        return None

    def get_graph_uri(self, wood_type: str) -> str:
        """
        Get the graph URI for a given wood type.

        Args:
            wood_type: The type of wood.

        Returns:
            The corresponding graph URI.
        """
        if wood_type in self.wood_type_graphs:
            return self.wood_type_graphs[wood_type]
        
        # Generate URI from template if not in predefined list
        return self.graph_uri_template.format(wood_type=wood_type.lower())

    def extract_property_type(self, query: str) -> Optional[str]:
        """
        Extract the property type being asked about in the query.

        Args:
            query: The user's natural language query.

        Returns:
            The detected property type, or None if not found.
        """
        query_lower = query.lower()
        
        for property_name, patterns in self.property_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    logger.debug(f"Detected property type: {property_name}")
                    return property_name
        
        return None

    def is_general_count_query(self, query: str) -> bool:
        """
        Detect if the query is asking about how many wood types/graphs are available.

        Args:
            query: The user's natural language query.

        Returns:
            True if the query is asking for a count of available wood types/graphs.
        """
        query_lower = query.lower()
        
        # If a specific wood type is mentioned, this is NOT a general count query
        if self.extract_wood_type(query) is not None:
            return False
        
        # Patterns that indicate count/list queries
        count_patterns = [
            "how many",
            "number of",
            "count of",
            "list all",
            "what are all",
            "show all",
            "available",
        ]
        
        # Subject patterns for wood types or graphs
        # More specific patterns to avoid false positives
        subject_patterns = [
            "wood type",
            "type of wood",
            "types of wood",
            "wood species",
            "species of wood",
            "graph",
            "dataset",
        ]
        
        has_count_pattern = any(pattern in query_lower for pattern in count_patterns)
        has_subject_pattern = any(pattern in query_lower for pattern in subject_patterns)
        
        return has_count_pattern and has_subject_pattern

    def analyze_query(self, query: str) -> dict:
        """
        Analyze a user query and extract relevant information.

        Args:
            query: The user's natural language query.

        Returns:
            A dictionary containing:
                - wood_type: The detected wood type (if any)
                - graph_uri: The corresponding graph URI (if wood type found)
                - property_type: The detected property type (if any)
                - predicate_filter: Suggested predicate filter for SPARQL query.
                                   Defaults to ["rdfs:subClassOf", "rdfs:hasValue"],
                                   overridden if property_type is detected.
                - is_general_count_query: Whether this is a query about available graphs/wood types
        """
        wood_type = self.extract_wood_type(query)
        property_type = self.extract_property_type(query)
        is_general_count = self.is_general_count_query(query)
        
        result = {
            "wood_type": wood_type,
            "graph_uri": self.get_graph_uri(wood_type) if wood_type else None,
            "property_type": property_type,
            "predicate_filter": ["rdfs:subClassOf", "rdfs:hasValue"],
            "search_terms": [],
            "is_general_count_query": is_general_count,
        }
        
        # Add predicate filter based on property type
        if property_type:
            result["predicate_filter"] = property_type.replace("_", " ")
            result["search_terms"].append(property_type.replace("_", " "))
        
        # Extract additional search terms from the query
        result["search_terms"].extend(self._extract_keywords(query))
        
        logger.info(f"Query analysis result: {result}")
        return result

    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract meaningful keywords from the query for searching.

        Args:
            query: The user's natural language query.

        Returns:
            A list of keywords.
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "what", "are", "the", "is", "a", "an", "of", "for", "in", "on",
            "to", "with", "how", "why", "when", "where", "which", "who",
            "can", "could", "would", "should", "does", "do", "did", "has",
            "have", "had", "been", "be", "being", "was", "were", "this",
            "that", "these", "those", "tree", "wood", "type", "tell", "me",
            "about", "give", "information", "please",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Remove wood types from keywords (already extracted)
        keywords = [w for w in keywords if w not in self.wood_type_graphs]
        
        return list(set(keywords))  # Remove duplicates

    def add_wood_type(self, wood_type: str, graph_uri: Optional[str] = None):
        """
        Add a new wood type to the analyzer.

        Args:
            wood_type: The name of the wood type.
            graph_uri: The corresponding graph URI. If not provided,
                       it will be generated from the template.
        """
        wood_type_lower = wood_type.lower()
        if graph_uri:
            self.wood_type_graphs[wood_type_lower] = graph_uri
        else:
            self.wood_type_graphs[wood_type_lower] = self.graph_uri_template.format(
                wood_type=wood_type_lower
            )
        
        # Compile regex pattern for the new wood type
        pattern = rf'\b{re.escape(wood_type_lower)}\b'
        self._wood_type_patterns[wood_type_lower] = re.compile(pattern, re.IGNORECASE)
