"""
GraphDB Connector Module.

Provides functionality for connecting to GraphDB (Ontotext) and executing SPARQL queries.
"""

import logging
from typing import Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


class GraphDBConnector:
    """
    A connector class for interacting with GraphDB (Ontotext) via SPARQL endpoint.
    
    Attributes:
        endpoint_url: The base URL of the GraphDB SPARQL endpoint.
        repository: The name of the repository to query.
        auth: Optional tuple of (username, password) for authentication.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:7200",
        repository: str = "Final_W2W_Onto",
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the GraphDB connector.

        Args:
            endpoint_url: Base URL of the GraphDB instance.
            repository: Name of the repository to query.
            username: Optional username for authentication.
            password: Optional password for authentication.
            timeout: Request timeout in seconds.
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.repository = repository
        self.auth = (username, password) if username and password else None
        self.timeout = timeout
        self.sparql_endpoint = f"{self.endpoint_url}/repositories/{self.repository}"

    def execute_sparql(self, query: str) -> list[dict]:
        """
        Execute a SPARQL query against GraphDB.

        Args:
            query: The SPARQL query string to execute.

        Returns:
            A list of dictionaries containing the query results.

        Raises:
            requests.RequestException: If the request to GraphDB fails.
        """
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(
                self.sparql_endpoint,
                data={"query": query},
                headers=headers,
                auth=self.auth,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return self._parse_sparql_results(result)
        except requests.RequestException as e:
            logger.error(f"SPARQL query execution failed: {e}")
            raise

    def _parse_sparql_results(self, result: dict) -> list[dict]:
        """
        Parse SPARQL JSON results into a list of dictionaries.

        Args:
            result: The raw JSON response from GraphDB.

        Returns:
            A list of dictionaries with variable bindings.
        """
        bindings = result.get("results", {}).get("bindings", [])
        parsed = []
        for binding in bindings:
            row = {}
            for var, value in binding.items():
                row[var] = value.get("value", "")
            parsed.append(row)
        return parsed

    def get_triples_from_graph(
        self,
        graph_uri: str,
        predicate_filter: Optional[str | list[str]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retrieve triples from a specific named graph.

        Args:
            graph_uri: The URI of the named graph to query.
            predicate_filter: Optional filter string or list of strings to match predicates.
                             If a list is provided, predicates matching any of the filters will be included.
            limit: Maximum number of results to return.

        Returns:
            A list of dictionaries containing subject, predicate, and object.
        """
        filter_clause = ""
        if predicate_filter:
            # Handle both single string and list of strings
            if isinstance(predicate_filter, str):
                # Single filter: use CONTAINS
                filter_clause = f'FILTER(CONTAINS(LCASE(STR(?p)), LCASE("{predicate_filter}")))'
            elif isinstance(predicate_filter, list) and predicate_filter:
                # Multiple filters: use OR condition with CONTAINS
                conditions = [f'CONTAINS(LCASE(STR(?p)), LCASE("{f}"))' for f in predicate_filter]
                filter_clause = f'FILTER({" || ".join(conditions)})'

        query = f"""
        SELECT ?s ?p ?o
        FROM <{graph_uri}>
        WHERE {{
            ?s ?p ?o .
            {filter_clause}
        }}
        LIMIT {limit}
        """
        return self.execute_sparql(query)

    def get_values_by_predicate(
        self,
        graph_uri: str,
        predicate_contains: str = "hasValue",
        subject_filter: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retrieve object values for triples matching a predicate pattern.

        This is useful for retrieving property values where predicates
        follow patterns like "rdfs:hasValue".

        Args:
            graph_uri: The URI of the named graph to query.
            predicate_contains: String that the predicate should contain.
            subject_filter: Optional filter to match specific subjects.
            limit: Maximum number of results to return.

        Returns:
            A list of dictionaries containing subject and object values.
        """
        subject_clause = ""
        if subject_filter:
            subject_clause = f'FILTER(CONTAINS(LCASE(STR(?s)), LCASE("{subject_filter}")))'

        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?s ?p ?o
        FROM <{graph_uri}>
        WHERE {{
            ?s ?p ?o .
            FILTER(CONTAINS(LCASE(STR(?p)), LCASE("{predicate_contains}")))
            {subject_clause}
        }}
        LIMIT {limit}
        """
        return self.execute_sparql(query)

    def get_triples_by_subject(
        self,
        graph_uri: str,
        subject_contains: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retrieve all triples where the subject contains a specific string.
        
        This retrieves ALL predicates and objects for matching subjects,
        ensuring complete information retrieval.

        Args:
            graph_uri: The URI of the named graph to query.
            subject_contains: String that the subject should contain.
            limit: Maximum number of results to return.

        Returns:
            A list of dictionaries containing subject, predicate, and object.
        """
        # Validate limit parameter
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        
        # Sanitize input to prevent SPARQL injection
        # Escape backslashes and quotes
        sanitized_subject = subject_contains.replace("\\", "\\\\").replace('"', '\\"')
        
        # Graph URIs should be valid URIs - basic validation
        if not graph_uri or not isinstance(graph_uri, str):
            raise ValueError("graph_uri must be a non-empty string")
        
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        
        SELECT ?s ?p ?o
        FROM <{graph_uri}>
        WHERE {{
            ?s ?p ?o .
            FILTER(CONTAINS(LCASE(STR(?s)), LCASE("{sanitized_subject}")))
        }}
        LIMIT {limit}
        """
        return self.execute_sparql(query)

    def list_named_graphs(self) -> list[str]:
        """
        List all named graphs in the repository.

        Returns:
            A list of graph URIs.
        """
        query = """
        SELECT DISTINCT ?g
        WHERE {
            GRAPH ?g { ?s ?p ?o }
        }
        """
        results = self.execute_sparql(query)
        return [r["g"] for r in results]

    def search_triples(
        self,
        graph_uri: str,
        search_term: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Search for triples where subject, predicate, or object contains the search term.

        Args:
            graph_uri: The URI of the named graph to query.
            search_term: The term to search for.
            limit: Maximum number of results to return.

        Returns:
            A list of matching triples.
        """
        query = f"""
        SELECT ?s ?p ?o
        FROM <{graph_uri}>
        WHERE {{
            ?s ?p ?o .
            FILTER(
                CONTAINS(LCASE(STR(?s)), LCASE("{search_term}")) ||
                CONTAINS(LCASE(STR(?p)), LCASE("{search_term}")) ||
                CONTAINS(LCASE(STR(?o)), LCASE("{search_term}"))
            )
        }}
        LIMIT {limit}
        """
        return self.execute_sparql(query)
