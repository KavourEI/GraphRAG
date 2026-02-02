"""
Tests for the GraphRAG package.

These tests focus on unit testing the components that don't require
external services (GraphDB, Ollama).
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from graphrag.query_analyzer import QueryAnalyzer
from graphrag.ollama_client import OllamaClient
from graphrag.graphdb_connector import GraphDBConnector


class TestGraphDBConnector(unittest.TestCase):
    """Tests for the GraphDBConnector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.connector = GraphDBConnector(
            endpoint_url="http://localhost:7200",
            repository="test_repo",
        )

    @patch.object(GraphDBConnector, 'execute_sparql')
    def test_get_triples_by_subject(self, mock_execute):
        """Test get_triples_by_subject method."""
        # Mock the response
        mock_execute.return_value = [
            {
                's': 'http://example.com/Traceability_of_parts',
                'p': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                'o': 'http://www.w3.org/2002/07/owl#Class'
            },
            {
                's': 'http://example.com/Traceability_of_parts',
                'p': 'http://www.w3.org/2000/01/rdf-schema#hasValue',
                'o': 'Plantation teak is traceable.'
            },
            {
                's': 'http://example.com/Traceability_of_parts',
                'p': 'http://www.w3.org/2000/01/rdf-schema#subClassOf',
                'o': 'http://example.com/End_of_Life'
            }
        ]
        
        # Call the method
        result = self.connector.get_triples_by_subject(
            graph_uri="http://example.com/graph",
            subject_contains="Traceability_of_parts",
        )
        
        # Verify the results
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['s'], 'http://example.com/Traceability_of_parts')
        
        # Verify the SPARQL query was called
        mock_execute.assert_called_once()
        
        # Verify query contains expected filters
        call_args = mock_execute.call_args[0][0]
        self.assertIn('FILTER(CONTAINS(LCASE(STR(?s)), LCASE("Traceability_of_parts")))', call_args)
        self.assertIn('SELECT ?s ?p ?o', call_args)

    @patch.object(GraphDBConnector, 'execute_sparql')
    def test_get_triples_by_subject_with_limit(self, mock_execute):
        """Test get_triples_by_subject method with custom limit."""
        mock_execute.return_value = []
        
        # Call the method with custom limit
        self.connector.get_triples_by_subject(
            graph_uri="http://example.com/graph",
            subject_contains="test_subject",
            limit=50,
        )
        
        # Verify the SPARQL query contains the limit
        call_args = mock_execute.call_args[0][0]
        self.assertIn('LIMIT 50', call_args)

    @patch.object(GraphDBConnector, 'execute_sparql')
    def test_get_triples_by_subject_sanitizes_input(self, mock_execute):
        """Test that get_triples_by_subject sanitizes input to prevent SPARQL injection."""
        mock_execute.return_value = []
        
        # Call the method with potentially dangerous input
        self.connector.get_triples_by_subject(
            graph_uri="http://example.com/graph",
            subject_contains='test"subject',
        )
        
        # Verify the SPARQL query has escaped quotes
        call_args = mock_execute.call_args[0][0]
        self.assertIn('test\\"subject', call_args)
        self.assertNotIn('test"subject', call_args.replace('test\\"subject', ''))

    def test_get_triples_by_subject_validates_limit(self):
        """Test that get_triples_by_subject validates limit parameter."""
        # Test with negative limit
        with self.assertRaises(ValueError):
            self.connector.get_triples_by_subject(
                graph_uri="http://example.com/graph",
                subject_contains="test",
                limit=-1,
            )
        
        # Test with zero limit
        with self.assertRaises(ValueError):
            self.connector.get_triples_by_subject(
                graph_uri="http://example.com/graph",
                subject_contains="test",
                limit=0,
            )

    def test_get_triples_by_subject_validates_graph_uri(self):
        """Test that get_triples_by_subject validates graph_uri parameter."""
        # Test with empty graph_uri
        with self.assertRaises(ValueError):
            self.connector.get_triples_by_subject(
                graph_uri="",
                subject_contains="test",
            )

    @patch.object(GraphDBConnector, 'execute_sparql')
    def test_get_triples_from_graph_with_single_filter(self, mock_execute):
        """Test get_triples_from_graph with a single predicate filter."""
        mock_execute.return_value = []
        
        # Call the method with a single filter string
        self.connector.get_triples_from_graph(
            graph_uri="http://example.com/graph",
            predicate_filter="rdfs:subClassOf",
        )
        
        # Verify the SPARQL query contains the filter
        call_args = mock_execute.call_args[0][0]
        self.assertIn('FILTER(CONTAINS(LCASE(STR(?p)), LCASE("rdfs:subClassOf")))', call_args)

    @patch.object(GraphDBConnector, 'execute_sparql')
    def test_get_triples_from_graph_with_list_filter(self, mock_execute):
        """Test get_triples_from_graph with multiple predicate filters."""
        mock_execute.return_value = []
        
        # Call the method with a list of filters
        self.connector.get_triples_from_graph(
            graph_uri="http://example.com/graph",
            predicate_filter=["rdfs:subClassOf", "rdfs:hasValue"],
        )
        
        # Verify the SPARQL query contains both filters with OR condition
        call_args = mock_execute.call_args[0][0]
        self.assertIn('FILTER(CONTAINS(LCASE(STR(?p)), LCASE("rdfs:subClassOf")) || CONTAINS(LCASE(STR(?p)), LCASE("rdfs:hasValue")))', call_args)

    @patch.object(GraphDBConnector, 'execute_sparql')
    def test_get_triples_from_graph_no_filter(self, mock_execute):
        """Test get_triples_from_graph with no predicate filter."""
        mock_execute.return_value = []
        
        # Call the method without a filter
        self.connector.get_triples_from_graph(
            graph_uri="http://example.com/graph",
        )
        
        # Verify the SPARQL query does not contain a FILTER clause
        call_args = mock_execute.call_args[0][0]
        self.assertNotIn('FILTER', call_args)


class TestQueryAnalyzer(unittest.TestCase):
    """Tests for the QueryAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = QueryAnalyzer()

    def test_extract_wood_type_oak(self):
        """Test extraction of 'oak' wood type."""
        result = self.analyzer.extract_wood_type("What are the properties of oak tree?")
        self.assertEqual(result, "oak")

    def test_extract_wood_type_pine(self):
        """Test extraction of 'pine' wood type."""
        result = self.analyzer.extract_wood_type("Tell me about pine wood")
        self.assertEqual(result, "pine")

    def test_extract_wood_type_maple(self):
        """Test extraction of 'maple' wood type."""
        result = self.analyzer.extract_wood_type("What is the color of maple?")
        self.assertEqual(result, "maple")

    def test_extract_wood_type_not_found(self):
        """Test extraction when no wood type is mentioned."""
        result = self.analyzer.extract_wood_type("What are wood properties?")
        self.assertIsNone(result)

    def test_get_graph_uri(self):
        """Test graph URI generation."""
        result = self.analyzer.get_graph_uri("oak")
        self.assertEqual(result, "http://w2w_onto.com/init/oak")

    def test_get_graph_uri_custom(self):
        """Test graph URI generation for custom wood type."""
        result = self.analyzer.get_graph_uri("custom_wood")
        self.assertEqual(result, "http://w2w_onto.com/init/custom_wood")

    def test_extract_property_type_material(self):
        """Test extraction of material properties."""
        result = self.analyzer.extract_property_type("material properties of wood")
        self.assertEqual(result, "material_properties")

    def test_extract_property_type_color(self):
        """Test extraction of color property."""
        result = self.analyzer.extract_property_type("What color is this wood?")
        self.assertEqual(result, "color")

    def test_extract_property_type_usage(self):
        """Test extraction of usage property."""
        result = self.analyzer.extract_property_type("What can I use this for?")
        self.assertEqual(result, "usage")

    def test_analyze_query_complete(self):
        """Test complete query analysis."""
        result = self.analyzer.analyze_query("What are the material properties of oak tree?")
        
        self.assertEqual(result["wood_type"], "oak")
        self.assertEqual(result["graph_uri"], "http://w2w_onto.com/init/oak")
        self.assertEqual(result["property_type"], "material_properties")
        # When property_type is detected, predicate_filter is set to the property type
        self.assertEqual(result["predicate_filter"], "material properties")

    def test_analyze_query_default_predicate_filter(self):
        """Test that predicate_filter defaults to rdfs:subClassOf and rdfs:hasValue."""
        result = self.analyzer.analyze_query("Tell me about oak")
        
        self.assertEqual(result["wood_type"], "oak")
        self.assertEqual(result["graph_uri"], "http://w2w_onto.com/init/oak")
        self.assertIsNone(result["property_type"])
        # Default predicate filter should be set
        self.assertEqual(result["predicate_filter"], ["rdfs:subClassOf", "rdfs:hasValue"])

    def test_analyze_query_no_wood_type(self):
        """Test query analysis when no wood type is found."""
        result = self.analyzer.analyze_query("What are wood properties?")
        
        self.assertIsNone(result["wood_type"])
        self.assertIsNone(result["graph_uri"])
        # Default predicate filter should still be set
        self.assertEqual(result["predicate_filter"], ["rdfs:subClassOf", "rdfs:hasValue"])

    def test_add_wood_type(self):
        """Test adding a new wood type."""
        self.analyzer.add_wood_type("redwood", "http://example.com/redwood")
        result = self.analyzer.extract_wood_type("Tell me about redwood")
        self.assertEqual(result, "redwood")

    def test_add_wood_type_auto_uri(self):
        """Test adding a wood type with auto-generated URI."""
        self.analyzer.add_wood_type("bamboo")
        result = self.analyzer.get_graph_uri("bamboo")
        self.assertEqual(result, "http://w2w_onto.com/init/bamboo")


class TestOllamaClientUtils(unittest.TestCase):
    """Tests for OllamaClient utility methods (no server required)."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = OllamaClient()

    def test_compute_similarity_identical(self):
        """Test similarity of identical vectors."""
        embedding = [1.0, 0.0, 0.0, 0.0]
        result = self.client.compute_similarity(embedding, embedding)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_compute_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors."""
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0, 0.0]
        result = self.client.compute_similarity(embedding1, embedding2)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_compute_similarity_opposite(self):
        """Test similarity of opposite vectors."""
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [-1.0, 0.0, 0.0, 0.0]
        result = self.client.compute_similarity(embedding1, embedding2)
        self.assertAlmostEqual(result, -1.0, places=5)

    def test_compute_similarity_empty(self):
        """Test similarity with empty vectors."""
        result = self.client.compute_similarity([], [])
        self.assertEqual(result, 0.0)

    def test_find_most_similar(self):
        """Test finding most similar embeddings."""
        query = [1.0, 0.0, 0.0]
        candidates = [
            [0.9, 0.1, 0.0],  # Most similar
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.5, 0.5, 0.0],  # Somewhat similar
        ]
        
        result = self.client.find_most_similar(query, candidates, top_k=2)
        
        self.assertEqual(len(result), 2)
        # First result should be index 0 (most similar)
        self.assertEqual(result[0][0], 0)


if __name__ == "__main__":
    unittest.main()
