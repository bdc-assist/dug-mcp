#!/usr/bin/env python3
"""
Redis Graph MCP Server
Provides Model Context Protocol interface to query biomedical knowledge graph stored in Redis
"""

import os
import asyncio
import argparse
import redis
from redis.commands.graph import Graph

import json
import re
import httpx
from reasoner_transpiler.cypher import get_query as trapi_get_query
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    Tool, TextContent,
    Prompt, PromptArgument, PromptMessage, GetPromptResult,
)
from dotenv import load_dotenv
import uvicorn

NAME_RESOLUTION_URL = "https://name-resolution-sri.renci.org/lookup"
SAP_QDRANT_URL = "https://sap-qdrant.apps.renci.org/annotate/"
PICSURE_SEARCH_URL = "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure/search/ac004461-1b47-4832-80e2-22a4aecabe39"

# Load environment variables from .env file
load_dotenv()
# Initialize MCP server
app = Server("redis-graph-server")

# Redis connection pool (initialized on first use)
redis_client = None
graph_db = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_synonym_where(curies: list, labels: list, fallback_term: str) -> str:
    """Build a Cypher WHERE expression from synonym enrichment results.

    Matches concepts by their CURIE id, their label name, or — when both lists
    are empty — falls back to a substring match on the raw search term.
    """
    clauses = []
    if curies:
        curie_list = ", ".join(f'"{c}"' for c in curies)
        clauses.append(f"concept.id IN [{curie_list}]")
    if labels:
        label_list = ", ".join(f'"{l}"' for l in labels)
        clauses.append(f"concept.name IN [{label_list}]")
    if not clauses:
        clauses.append(f"concept.name CONTAINS '{fallback_term}'")
    return " OR ".join(clauses)


def _truncate(out: str, limit: int = 50_000) -> str:
    """Trim a JSON string to *limit* characters and append a notice if trimmed."""
    if len(out) > limit:
        return out[:limit] + "\n... (truncated)"
    return out


async def fetch_synonyms(search_term: str) -> dict:
    """
    Enrich a search term by fetching synonyms and related concept identifiers from:
    - Name Resolution SRI: returns normalized CURIEs and preferred labels
    - SAP-BERT Qdrant: returns semantically similar biomedical concepts

    Returns {'curies': [...], 'labels': [...], 'warnings': [...]}
    Warnings are populated when a service is unreachable or returns an error,
    so callers can signal degraded results to the user.
    """
    curies = []
    labels = []
    warnings = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        nr_coro = client.get(NAME_RESOLUTION_URL, params={"string": search_term, "limit": 10})
        sap_coro = client.post(
            SAP_QDRANT_URL,
            json={"text": search_term, "model_name": "sapbert", "count": 10, "args": {}}
        )
        nr_resp, sap_resp = await asyncio.gather(nr_coro, sap_coro, return_exceptions=True)

    # Name Resolution SRI: [{curie, label, synonyms, ...}, ...]
    if isinstance(nr_resp, Exception):
        warnings.append(f"Name Resolution SRI unavailable: {nr_resp}")
    elif nr_resp.status_code != 200:
        warnings.append(f"Name Resolution SRI returned HTTP {nr_resp.status_code} — results may be incomplete")
    else:
        try:
            for item in nr_resp.json()[:10]:
                if curie := item.get("curie"):
                    curies.append(curie)
                if label := item.get("label"):
                    labels.append(label)
        except Exception as e:
            warnings.append(f"Name Resolution SRI response could not be parsed: {e}")

    # SAP-BERT Qdrant: response structure varies; handle list or wrapped dict
    if isinstance(sap_resp, Exception):
        warnings.append(f"SAP-BERT unavailable: {sap_resp}")
    elif sap_resp.status_code != 200:
        warnings.append(f"SAP-BERT returned HTTP {sap_resp.status_code} — results may be incomplete")
    else:
        try:
            sap_data = sap_resp.json()
            items = sap_data if isinstance(sap_data, list) else (
                sap_data.get("results") or sap_data.get("hits") or
                sap_data.get("annotations") or sap_data.get("concepts") or []
            )
            for item in items[:10]:
                if not isinstance(item, dict):
                    continue
                for id_field in ("id", "curie", "identifier", "concept_id"):
                    if val := item.get(id_field):
                        curies.append(val)
                        break
                for name_field in ("name", "label", "text", "concept_name"):
                    if val := item.get(name_field):
                        labels.append(val)
                        break
        except Exception as e:
            warnings.append(f"SAP-BERT response could not be parsed: {e}")

    return {
        "curies": list(dict.fromkeys(curies)),   # deduplicate, preserve order
        "labels": list(dict.fromkeys(labels)),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------

def get_redis_connection():
    """Establish Redis connection with authentication.

    Uses a local variable for the Redis client so that globals are only updated
    after a successful ping — preventing a sticky-failure state where redis_client
    is set but graph_db is still None on every subsequent call.
    """
    global redis_client, graph_db

    if redis_client is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        graph_name = os.getenv("REDIS_GRAPH_NAME", "test")

        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        client.ping()  # raises on failure; globals are NOT updated if this throws
        redis_client = client
        graph_db = Graph(redis_client, graph_name)

    return redis_client, graph_db


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def _serialize_value(val):
    """Convert a RedisGraph Node/Edge/Path to a clean dict; pass through scalars."""
    cls = type(val).__name__
    if cls == "Node":
        # labels is a list ordered general→specific; pick the last (most specific)
        labels = val.labels if val.labels else []
        label = labels[-1] if labels else ""
        category = label.replace("biolink.", "biolink:") if label else None
        props = dict(val.properties) if val.properties else {}
        out = {}
        if category:
            out["category"] = category
        out["id"] = props.get("id", props.get("curie", None))
        out["name"] = props.get("name", None)
        extra = {k: v for k, v in props.items() if k not in ("id", "curie", "name")}
        if extra:
            out["properties"] = extra
        return out
    elif cls == "Edge":
        rel = getattr(val, "relation", None) or ""
        predicate = rel.replace("biolink.", "biolink:") if rel else None
        props = dict(val.properties) if val.properties else {}
        return {
            "predicate": predicate,
            "src_node": val.src_node,
            "dest_node": val.dest_node,
            **({"properties": props} if props else {}),
        }
    elif cls == "Path":
        # Path.nodes() and Path.edges() are methods, not properties
        nodes = val.nodes() if callable(val.nodes) else val.nodes
        edges = val.edges() if callable(val.edges) else val.edges
        return {"nodes": [_serialize_value(n) for n in nodes],
                "edges": [_serialize_value(e) for e in edges]}
    return val


def results_to_list(result_set, header) -> list[dict]:
    """Convert a RedisGraph result set to a list of dicts keyed by column name."""
    if not result_set or not header:
        return []
    col_names = [col[1] for col in header]
    return [{col: _serialize_value(val) for col, val in zip(col_names, row)} for row in result_set]


def to_json(data) -> str:
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """Expose reusable workflow prompts for chatbots and LLM clients."""
    return [
        Prompt(
            name="find_variables_for_concept",
            description=(
                "Find study variables in the BDC knowledge graph related to a biomedical concept. "
                "Enriches the concept with synonyms from Name Resolution SRI and SAP-BERT, "
                "then searches for connected study variables."
            ),
            arguments=[
                PromptArgument(
                    name="concept",
                    description="Biomedical concept to search for (e.g. 'asthma', 'diabetes', 'cholesterol')",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="explore_concept",
            description=(
                "Get a full picture of a concept in the knowledge graph: what entities are connected to it, "
                "which studies have measured it, and what variables are associated with it."
            ),
            arguments=[
                PromptArgument(
                    name="concept_id",
                    description="Concept CURIE identifier (e.g. 'MONDO:0004979' for asthma)",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="find_studies_for_disease",
            description=(
                "End-to-end workflow: given a disease or phenotype name, find all studies in the BDC "
                "knowledge graph that have collected data related to it, via synonym-enriched concept lookup."
            ),
            arguments=[
                PromptArgument(
                    name="disease_name",
                    description="Disease or phenotype name (e.g. 'asthma', 'hypertension')",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="explain_variable",
            description=(
                "Explain what a study variable measures and its full biomedical context: "
                "which concepts it is linked to and which studies it belongs to."
            ),
            arguments=[
                PromptArgument(
                    name="variable_id",
                    description="Study variable identifier (e.g. 'phv00430345.v1.p1')",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="find_path_between_concepts",
            description=(
                "Discover how two biomedical concepts are related to each other through the knowledge graph "
                "and explain the connection path in plain language."
            ),
            arguments=[
                PromptArgument(
                    name="concept_a_id",
                    description="First concept CURIE (e.g. 'MONDO:0004979')",
                    required=True,
                ),
                PromptArgument(
                    name="concept_b_id",
                    description="Second concept CURIE (e.g. 'MONDO:0004784')",
                    required=True,
                ),
            ],
        ),
        Prompt(
            name="trapi_query_builder",
            description=(
                "Build and execute a TRAPI query graph against the knowledge graph. "
                "Guides the LLM to construct a valid TRAPI query for a given biomedical question, "
                "with automatic biolink predicate hierarchy expansion."
            ),
            arguments=[
                PromptArgument(
                    name="question",
                    description="Biomedical question to answer (e.g. 'What study variables are associated with asthma?')",
                    required=True,
                ),
            ],
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    """Return the message template for the requested prompt."""

    if name == "find_variables_for_concept":
        concept = arguments.get("concept", "")
        return GetPromptResult(
            description=f"Find study variables related to '{concept}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"I want to find study variables in the BDC biomedical knowledge graph "
                            f"related to the concept '{concept}'.\n\n"
                            f"Please use the `search_concepts` tool with:\n"
                            f"  - search_term: \"{concept}\"\n"
                            f"  - find_variables: true\n\n"
                            f"This will automatically enrich '{concept}' with synonyms from "
                            f"Name Resolution SRI and SAP-BERT before searching.\n\n"
                            f"Once you have the results, summarize:\n"
                            f"1. How many synonyms were found and what they are\n"
                            f"2. How many study variables were matched\n"
                            f"3. The most relevant variables and which concepts linked them"
                        ),
                    ),
                )
            ],
        )

    elif name == "explore_concept":
        concept_id = arguments.get("concept_id", "")
        return GetPromptResult(
            description=f"Explore all connections for concept '{concept_id}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Explore the biomedical concept '{concept_id}' in the knowledge graph.\n\n"
                            f"Please run these tools in order:\n"
                            f"1. `get_concept_connections` with concept_id='{concept_id}' — to see all directly connected entities\n"
                            f"2. `find_connected_studies` with concept_id='{concept_id}' — to find studies that have measured this concept\n"
                            f"3. `get_concept_graph` with concept_id='{concept_id}' — for the full subgraph\n\n"
                            f"Then provide a clear summary:\n"
                            f"- What is this concept and what types of entities are connected to it?\n"
                            f"- How many study variables and studies are linked?\n"
                            f"- What are the most notable connections?"
                        ),
                    ),
                )
            ],
        )

    elif name == "find_studies_for_disease":
        disease_name = arguments.get("disease_name", "")
        return GetPromptResult(
            description=f"Find studies that collected data on '{disease_name}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"I want to find all studies in the BDC knowledge graph that have collected "
                            f"data related to '{disease_name}'.\n\n"
                            f"Step 1: Use `search_concepts` with search_term='{disease_name}' and find_variables=true "
                            f"to get synonym-enriched study variables.\n\n"
                            f"Step 2: For each unique concept_id returned, use `find_connected_studies` "
                            f"to get the full list of studies.\n\n"
                            f"Finally, summarize:\n"
                            f"- Which synonyms were matched in the graph\n"
                            f"- The complete list of studies found\n"
                            f"- How many unique variables and studies are available for '{disease_name}'"
                        ),
                    ),
                )
            ],
        )

    elif name == "explain_variable":
        variable_id = arguments.get("variable_id", "")
        return GetPromptResult(
            description=f"Explain study variable '{variable_id}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Please explain the study variable '{variable_id}' in plain language.\n\n"
                            f"Use `get_variable_details` with variable_id='{variable_id}' to retrieve its details.\n\n"
                            f"Then explain:\n"
                            f"1. What does this variable measure or represent?\n"
                            f"2. Which biomedical concepts is it linked to (diseases, phenotypes, procedures)?\n"
                            f"3. Which study does it belong to?\n"
                            f"4. How might a researcher use this variable in a study?"
                        ),
                    ),
                )
            ],
        )

    elif name == "find_path_between_concepts":
        concept_a = arguments.get("concept_a_id", "")
        concept_b = arguments.get("concept_b_id", "")
        return GetPromptResult(
            description=f"Find relationship path between '{concept_a}' and '{concept_b}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Discover how the concepts '{concept_a}' and '{concept_b}' are related "
                            f"in the biomedical knowledge graph.\n\n"
                            f"Use `find_concept_paths` with source_id='{concept_a}' and target_id='{concept_b}'.\n\n"
                            f"If no direct path is found, also try `expand_concept` on each to find "
                            f"shared neighbors.\n\n"
                            f"Then explain in plain language:\n"
                            f"1. Are these concepts directly or indirectly related?\n"
                            f"2. What is the relationship path between them?\n"
                            f"3. What does this connection mean biomedically?"
                        ),
                    ),
                )
            ],
        )

    elif name == "trapi_query_builder":
        question = arguments.get("question", "")
        return GetPromptResult(
            description=f"Build a TRAPI query to answer: '{question}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"I want to answer this biomedical question using the knowledge graph:\n\n"
                            f"\"{question}\"\n\n"
                            f"Please construct a TRAPI query graph and run it using the `trapi_query` tool.\n\n"
                            f"A TRAPI query graph has this structure:\n"
                            f"{{\n"
                            f"  \"nodes\": {{\n"
                            f"    \"<node_key>\": {{\"ids\": [\"<CURIE>\"], \"categories\": [\"biolink:<Type>\"]}}\n"
                            f"  }},\n"
                            f"  \"edges\": {{\n"
                            f"    \"<edge_key>\": {{\"subject\": \"<node_key>\", \"object\": \"<node_key>\", \"predicates\": [\"biolink:<predicate>\"]}}\n"
                            f"  }}\n"
                            f"}}\n\n"
                            f"Guidelines:\n"
                            f"- Use biolink categories like: biolink:Disease, biolink:PhenotypicFeature, biolink:StudyVariable, biolink:Gene\n"
                            f"- Use biolink predicates like: biolink:related_to, biolink:has_phenotype, biolink:associated_with\n"
                            f"- Omit 'ids' or 'categories' if unknown — the graph will match any node\n"
                            f"- Omit 'predicates' on an edge to match any relationship\n"
                            f"- The predicate hierarchy is automatically expanded (e.g. biolink:related_to includes all child predicates)\n\n"
                            f"If you're unsure of the exact CURIE, first use `search_concepts` to find it, "
                            f"then build the TRAPI query with the ID."
                        ),
                    ),
                )
            ],
        )

    else:
        raise ValueError(f"Unknown prompt: {name}")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for querying the Redis graph."""
    return [
        Tool(
            name="trapi_query",
            description=(
                "Query the knowledge graph using a standard TRAPI (Translator Reasoner API) query graph. "
                "Automatically handles the biolink predicate hierarchy — querying a parent predicate like "
                "'biolink:related_to' also matches all its child predicates (e.g. associated_with, correlated_with, etc.). "
                "Use this for structured biomedical queries without needing to know Cypher syntax. "
                "Nodes can be filtered by biolink category and/or specific IDs (CURIEs). "
                "Edges can be filtered by biolink predicate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "qgraph": {
                        "type": "object",
                        "description": (
                            "TRAPI query graph with 'nodes' (dict of node_key → {ids?, categories?}) "
                            "and 'edges' (dict of edge_key → {subject, object, predicates?}). "
                            "Example: {\"nodes\": {\"disease\": {\"ids\": [\"MONDO:0004979\"], \"categories\": [\"biolink:Disease\"]}, "
                            "\"variable\": {\"categories\": [\"biolink:StudyVariable\"]}}, "
                            "\"edges\": {\"e0\": {\"subject\": \"disease\", \"object\": \"variable\"}}}"
                        ),
                        "properties": {
                            "nodes": {"type": "object"},
                            "edges": {"type": "object"}
                        },
                        "required": ["nodes", "edges"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 50
                    }
                },
                "required": ["qgraph"]
            }
        ),
        Tool(
            name="cypher_query",
            description="Execute a raw Cypher query on the Redis biomedical knowledge graph. Use backticks for labels with dots: `biolink.Disease`",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Cypher query to execute. Remember to escape biolink labels with backticks: (n:`biolink.Disease`)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Override LIMIT clause (optional)",
                        "default": 50
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_concepts",
            description=(
                "Search for biomedical concepts by name, OR find StudyVariables related to a concept with synonym enrichment. "
                "When find_variables=true, the search term is first expanded into synonyms and related concept identifiers "
                "by querying two external services in parallel: "
                "(1) Name Resolution SRI (https://name-resolution-sri.renci.org) — returns up to 10 normalized CURIEs and preferred labels; "
                "(2) SAP-BERT Qdrant (https://sap-qdrant.example.com) — returns up to 10 semantically similar biomedical concepts. "
                "The combined set of CURIEs and labels is then used to search the Redis knowledge graph for any StudyVariables "
                "connected to those concepts, providing broader coverage than a simple name match."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Concept name or keyword to search for (e.g., 'cholesterol', 'heart disease', 'asthma')"
                    },
                    "node_type": {
                        "type": "string",
                        "description": "Optional: Filter by node type (Disease, PhenotypicFeature, Procedure, Cell, OrganismTaxon, etc.)",
                        "enum": ["Disease", "PhenotypicFeature", "Procedure", "Cell", "OrganismTaxon", "Study", "Publication", "BiologicalProcess"]
                    },
                    "find_variables": {
                        "type": "boolean",
                        "description": "Set to true to enrich the search term with synonyms via Name Resolution SRI and SAP-BERT, then find StudyVariables connected to those concepts in the Redis graph",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20
                    }
                },
                "required": ["search_term"]
            }
        ),
        Tool(
            name="get_concept_graph",
            description="Get the complete subgraph for a biomedical concept including studies, variables, and related entities",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_id": {
                        "type": "string",
                        "description": "The concept ID (e.g., MONDO:0005015 for disease, phv00283870.v1.p1 for variable)"
                    },
                    "expand_depth": {
                        "type": "integer",
                        "description": "How many hops to follow (1-3)",
                        "default": 2
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum nodes to return",
                        "default": 100
                    }
                },
                "required": ["concept_id"]
            }
        ),
        Tool(
            name="get_concept_connections",
            description="List all entities directly connected to a concept, showing the relationship type, connected node type, name, and ID. Use node_type_filter to focus on a specific entity type (e.g. only StudyVariables or only Studies).",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_id": {
                        "type": "string",
                        "description": "Concept ID to analyze (works for any node: disease, phenotype, variable, etc.)"
                    },
                    "node_type_filter": {
                        "type": "string",
                        "description": "Optional: return only neighbors of this biolink type",
                        "enum": ["Disease", "PhenotypicFeature", "Procedure", "Cell", "OrganismTaxon", "Study", "StudyVariable", "BiologicalProcess"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of connections to return",
                        "default": 50
                    }
                },
                "required": ["concept_id"]
            }
        ),
        Tool(
            name="list_graph_schema",
            description="List all node types and relationships in the graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "show_counts": {
                        "type": "boolean",
                        "description": "Include count of each node type",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="find_highly_connected_variables",
            description="Find StudyVariables with the most connections to biomedical concepts",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_connections": {
                        "type": "integer",
                        "description": "Minimum number of connections",
                        "default": 10
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="search_variables_by_name",
            description=(
                "Search for StudyVariables by free-text name or by ID prefix/pattern. "
                "Unlike search_concepts, this searches only StudyVariable nodes and supports "
                "partial ID matching (e.g. 'phv00430' returns all variables whose ID starts with that prefix). "
                "Use this when you already know you want variables and have a name fragment or a partial phv ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Variable name fragment or ID prefix (e.g., 'asthma', 'phv00430', 'bmi')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20
                    }
                },
                "required": ["search_term"]
            }
        ),
        Tool(
            name="expand_concept",
            description="Expand a biomedical concept by finding related concepts, synonyms, and transitive connections up to N hops away",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_id": {
                        "type": "string",
                        "description": "Concept ID to expand (e.g., MONDO:0004784 for allergic asthma)"
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum number of hops/relationships to traverse (1-3)",
                        "default": 2
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by specific relationship types (e.g., ['related_to', 'part_of'])"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of expanded concepts to return",
                        "default": 50
                    }
                },
                "required": ["concept_id"]
            }
        ),
        Tool(
            name="find_concept_paths",
            description="Find paths between two biomedical concepts through the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source concept ID"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Target concept ID"
                    },
                    "max_path_length": {
                        "type": "integer",
                        "description": "Maximum path length to search (1-5)",
                        "default": 3
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of paths to return",
                        "default": 10
                    }
                },
                "required": ["source_id", "target_id"]
            }
        ),
        Tool(
            name="picsure_search",
            description=(
                "Look up study variables in BioData Catalyst PIC-SURE by phv ID or keyword. "
                "When semantic=true (default when keyword is provided), enriches the keyword via "
                "Name Resolution SRI + SAP-BERT, queries the knowledge graph for semantically linked "
                "variables, then looks up their PIC-SURE paths — finding variables related to "
                "'myocardial infarction', 'MI', etc. when you search 'heart attack'. "
                "When semantic=false, forwards the keyword directly to PIC-SURE (literal match only). "
                "No authentication required — uses the open PIC-SURE resource."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "phv_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of phv IDs to look up (e.g. ['phv00425822', 'phv00347788']). "
                            "Strip version suffixes — use 'phv00425822' not 'phv00425822.v1.p1'."
                        )
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Biomedical concept to search for (e.g. 'heart attack', 'asthma'). With semantic=true, uses KG synonym enrichment."
                    },
                    "semantic": {
                        "type": "boolean",
                        "description": "If true (default), enrich keyword via Name Resolution SRI + SAP-BERT and query the KG before PIC-SURE lookup. If false, forward keyword directly to PIC-SURE as a literal search.",
                        "default": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of variable paths to return",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="find_cohort_variables",
            description=(
                "Find study variables for multiple biomedical concepts simultaneously and identify "
                "studies that contain variables for ALL specified concepts — enabling cohort feasibility analysis. "
                "For each concept, synonym enrichment is used to find matching variables in the knowledge graph. "
                "Returns feasible_studies (have all concepts) and partial_studies (missing some), with "
                "PIC-SURE variable paths and a ready-to-submit PIC-SURE query template for the best study. "
                "Use this to build multi-condition cohort queries (e.g. 'asthma AND obesity')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of biomedical concepts to search for (e.g. ['asthma', 'obesity'])"
                    },
                    "require_all": {
                        "type": "boolean",
                        "description": "If true (default), only highlight studies that have variables for ALL concepts.",
                        "default": True
                    },
                    "variables_per_concept": {
                        "type": "integer",
                        "description": "Max variables to collect per concept from the KG",
                        "default": 20
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of feasible studies to return",
                        "default": 10
                    }
                },
                "required": ["concepts"]
            }
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers — one async function per tool
# ---------------------------------------------------------------------------

async def _handle_trapi_query(arguments: dict, graph) -> list[TextContent]:
    qgraph = arguments["qgraph"]
    limit  = arguments.get("limit", 50)

    # Generate Cypher from TRAPI query graph using reasoner-transpiler.
    # Use memgraph dialect (uses id() instead of elementId()) — closer to RedisGraph.
    # reasoner=False gives plain Cypher without TRAPI result wrapping.
    cypher = trapi_get_query(qgraph, reasoner=False, dialect="memgraph")

    # RedisGraph uses backtick-wrapped dot notation: `biolink.Disease`
    # Transpiler outputs colon notation: `biolink:Disease` — convert it.
    cypher = re.sub(r'`biolink:(\w+)`', r'`biolink.\1`', cypher)

    if "LIMIT" not in cypher.upper():
        cypher += f" LIMIT {limit}"

    result = graph.query(cypher)
    rows = results_to_list(result.result_set, result.header) if result.result_set else []
    return [TextContent(type="text", text=to_json({
        "qgraph": qgraph,
        "cypher_used": cypher,
        "total_results": len(rows),
        "results": rows,
    }))]


async def _handle_cypher_query(arguments: dict, graph) -> list[TextContent]:
    query = arguments["query"]
    result = graph.query(query)

    if result.result_set:
        rows = results_to_list(result.result_set, result.header)
        out = _truncate(to_json({
            "rows_returned": len(rows),
            "execution_time_ms": result.execution_time,
            "results": rows,
        }))
        return [TextContent(type="text", text=out)]
    else:
        return [TextContent(type="text", text=to_json({"rows_returned": 0, "results": []}))]


async def _handle_search_concepts(arguments: dict, graph) -> list[TextContent]:
    search_term    = arguments["search_term"]
    node_type      = arguments.get("node_type")
    find_variables = arguments.get("find_variables", False)
    limit          = arguments.get("limit", 20)

    if find_variables:
        synonyms = await fetch_synonyms(search_term)
        curies   = synonyms["curies"]
        labels   = synonyms["labels"]
        warnings = synonyms["warnings"]

        where_expr = _build_synonym_where(curies, labels, search_term)

        # Fetch all (variable, concept, predicate) matches — no LIMIT here.
        # Dedup is done in Python after grouping by variable_id.
        query = f"""
        MATCH (concept)-[r]-(v:`biolink.StudyVariable`)
        WHERE ({where_expr})
          AND NOT labels(concept)[0] = 'biolink.StudyVariable'
        RETURN
            v.id AS variable_id,
            v.name AS variable_name,
            v.description AS variable_description,
            concept.id AS concept_id,
            concept.name AS concept_name,
            labels(concept)[0] AS concept_type,
            type(r) AS predicate
        """
        result = graph.query(query)
        rows = results_to_list(result.result_set, result.header) if result.result_set else []

        # Group by variable_id — one entry per unique variable,
        # with a matched_concepts list showing which concepts and predicates matched.
        variables: dict = {}
        for row in rows:
            vid = row["variable_id"]
            if vid not in variables:
                variables[vid] = {
                    "variable_id": vid,
                    "variable_name": row["variable_name"],
                    "variable_description": row.get("variable_description"),
                    "matched_concepts": [],
                }
            predicate  = row["predicate"].replace("biolink.", "biolink:") if row.get("predicate") else None
            concept_id = row["concept_id"]
            # Dedup matched concepts by (concept_id, predicate) — graph can have
            # edges in both directions between the same pair of nodes.
            seen = {(c["concept_id"], c["predicate"]) for c in variables[vid]["matched_concepts"]}
            if (concept_id, predicate) not in seen:
                variables[vid]["matched_concepts"].append({
                    "concept_id": concept_id,
                    "concept_name": row["concept_name"],
                    "concept_type": row["concept_type"].replace("biolink.", "biolink:") if row.get("concept_type") else None,
                    "predicate": predicate,
                })

        # Sort by number of matched concepts descending (relevance proxy),
        # then apply limit on unique variables.
        unique_vars = sorted(variables.values(), key=lambda v: len(v["matched_concepts"]), reverse=True)
        unique_vars = unique_vars[:limit]

        out = _truncate(to_json({
            "search_term": search_term,
            "enrichment": {
                "curies": curies,
                "labels": labels,
                **({"warnings": warnings} if warnings else {}),
            },
            "total_results": len(unique_vars),
            "variables": unique_vars,
        }))
        return [TextContent(type="text", text=out)]

    else:
        type_match = f"(n:`biolink.{node_type}`)" if node_type else "(n)"
        query = f"""
        MATCH {type_match}
        WHERE n.name CONTAINS '{search_term}'
        RETURN labels(n)[0] AS type, n.name AS name, n.id AS id
        LIMIT {limit}
        """
        result = graph.query(query)
        rows = results_to_list(result.result_set, result.header) if result.result_set else []
        out = _truncate(to_json({
            "search_term": search_term,
            "node_type": node_type,
            "total_results": len(rows),
            "concepts": rows,
        }))
        return [TextContent(type="text", text=out)]


async def _handle_get_concept_graph(arguments: dict, graph) -> list[TextContent]:
    concept_id   = arguments["concept_id"]
    expand_depth = min(arguments.get("expand_depth", 2), 3)
    limit        = arguments.get("limit", 100)

    if expand_depth == 1:
        query = f"""
        MATCH (concept {{id: "{concept_id}"}})-[r1]-(connected)
        RETURN concept.name AS concept,
               type(r1) AS rel_type,
               labels(connected)[0] AS connected_type,
               connected.name AS connected_name,
               connected.id AS connected_id
        LIMIT {limit}
        """
    else:
        query = f"""
        MATCH (concept {{id: "{concept_id}"}})-[r1]-(variable:`biolink.StudyVariable`)
        OPTIONAL MATCH (variable)-[r2]-(study:`biolink.Study`)
        OPTIONAL MATCH (variable)-[r3]-(related)
        WHERE related <> concept
        RETURN DISTINCT
            concept.name AS concept,
            concept.id AS concept_id,
            labels(concept)[0] AS concept_type,
            variable.name AS variable_name,
            variable.id AS variable_id,
            study.name AS study_name,
            study.id AS study_id,
            COUNT(DISTINCT related) AS related_concepts_count
        LIMIT {limit}
        """

    result = graph.query(query)
    if result.result_set:
        rows = results_to_list(result.result_set, result.header)
        for row in rows:
            if row.get("concept_type"):
                row["concept_type"] = row["concept_type"].replace("biolink.", "biolink:")
        out = _truncate(to_json({
            "concept_id": concept_id,
            "expand_depth": expand_depth,
            "total_results": len(rows),
            "graph": rows,
        }))
        return [TextContent(type="text", text=out)]
    else:
        return [TextContent(type="text", text=to_json({"concept_id": concept_id, "total_results": 0, "graph": []}))]


async def _handle_get_concept_connections(arguments: dict, graph) -> list[TextContent]:
    concept_id      = arguments["concept_id"]
    node_type_filter = arguments.get("node_type_filter")
    limit           = arguments.get("limit", 50)

    type_clause = f":`biolink.{node_type_filter}`" if node_type_filter else ""

    summary_result = graph.query(f"""
    MATCH (concept {{id: "{concept_id}"}})-[r]-(connected{type_clause})
    WITH labels(connected)[0] AS entity_type, COUNT(*) AS count
    RETURN entity_type, count
    ORDER BY count DESC
    """)

    detail_result = graph.query(f"""
    MATCH (concept {{id: "{concept_id}"}})-[r]-(connected{type_clause})
    RETURN
        type(r) AS relationship,
        labels(connected)[0] AS connected_type,
        connected.name AS connected_name,
        connected.id AS connected_id
    ORDER BY labels(connected)[0], type(r), connected.name
    LIMIT {limit}
    """)

    summary     = [{"entity_type": row[0], "count": row[1]} for row in (summary_result.result_set or [])]
    connections = results_to_list(detail_result.result_set, detail_result.header) if detail_result.result_set else []
    out = _truncate(to_json({
        "concept_id": concept_id,
        "node_type_filter": node_type_filter,
        "summary": summary,
        "total_connections_shown": len(connections),
        "connections": connections,
    }))
    return [TextContent(type="text", text=out)]


async def _handle_list_graph_schema(arguments: dict, graph) -> list[TextContent]:
    show_counts = arguments.get("show_counts", True)

    if show_counts:
        query = """
        MATCH (n)
        WITH labels(n)[0] AS node_type, COUNT(*) AS count
        RETURN node_type, count
        ORDER BY count DESC
        """
    else:
        query = """
        MATCH (n)
        WITH DISTINCT labels(n)[0] AS node_type
        RETURN node_type
        ORDER BY node_type
        """

    result = graph.query(query)
    rows = results_to_list(result.result_set, result.header) if result.result_set else []
    return [TextContent(type="text", text=to_json({"schema": rows}))]


async def _handle_find_highly_connected_variables(arguments: dict, graph) -> list[TextContent]:
    min_connections = arguments.get("min_connections", 10)
    limit           = arguments.get("limit", 20)

    query = f"""
    MATCH (v:`biolink.StudyVariable`)--(c)
    WITH v, COUNT(DISTINCT c) AS connection_count
    WHERE connection_count >= {min_connections}
    RETURN v.name AS variable_name, v.id AS variable_id, connection_count
    ORDER BY connection_count DESC
    LIMIT {limit}
    """

    result = graph.query(query)
    rows = results_to_list(result.result_set, result.header) if result.result_set else []
    return [TextContent(type="text", text=to_json({
        "min_connections": min_connections,
        "total_results": len(rows),
        "variables": rows,
    }))]


async def _handle_search_variables_by_name(arguments: dict, graph) -> list[TextContent]:
    search_term = arguments["search_term"]
    limit       = arguments.get("limit", 20)

    query = f"""
    MATCH (v:`biolink.StudyVariable`)
    WHERE v.name CONTAINS '{search_term}' OR v.id CONTAINS '{search_term}'
    RETURN v.id AS variable_id, v.name AS variable_name
    LIMIT {limit}
    """

    result = graph.query(query)
    rows = results_to_list(result.result_set, result.header) if result.result_set else []
    out = _truncate(to_json({"search_term": search_term, "total_results": len(rows), "variables": rows}))
    return [TextContent(type="text", text=out)]


async def _handle_expand_concept(arguments: dict, graph) -> list[TextContent]:
    concept_id         = arguments["concept_id"]
    max_hops           = min(arguments.get("max_hops", 2), 3)
    relationship_types = arguments.get("relationship_types")
    limit              = arguments.get("limit", 50)

    # Conditionally add a relationship-type filter to the WHERE clause.
    rel_filter = (
        f"AND ALL(rel in relationships(path) WHERE type(rel) IN {relationship_types})"
        if relationship_types else ""
    )
    query = f"""
    MATCH path = (source {{id: "{concept_id}"}})-[r*1..{max_hops}]-(expanded)
    WHERE source <> expanded
      {rel_filter}
    WITH expanded,
         labels(expanded)[0] AS expanded_type,
         length(path) AS hops,
         [rel in relationships(path) | type(rel)] AS path_relationships
    RETURN DISTINCT
        expanded.id AS concept_id,
        expanded.name AS concept_name,
        expanded_type,
        hops,
        path_relationships
    ORDER BY hops, expanded.name
    LIMIT {limit}
    """

    result = graph.query(query)
    rows = results_to_list(result.result_set, result.header) if result.result_set else []
    out = _truncate(to_json({
        "concept_id": concept_id,
        "max_hops": max_hops,
        "relationship_types": relationship_types,
        "total_results": len(rows),
        "expanded": rows,
    }))
    return [TextContent(type="text", text=out)]


async def _handle_find_concept_paths(arguments: dict, graph) -> list[TextContent]:
    source_id       = arguments["source_id"]
    target_id       = arguments["target_id"]
    max_path_length = min(arguments.get("max_path_length", 3), 5)
    limit           = arguments.get("limit", 10)

    query = f"""
    MATCH path = shortestPath((source {{id: "{source_id}"}})-[*1..{max_path_length}]-(target {{id: "{target_id}"}}))
    WITH path, length(path) AS path_length
    UNWIND nodes(path) AS node
    UNWIND relationships(path) AS rel
    WITH path, path_length,
         collect(DISTINCT node.name) AS node_names,
         collect(DISTINCT type(rel)) AS relationship_types
    RETURN
        path_length,
        node_names,
        relationship_types
    ORDER BY path_length
    LIMIT {limit}
    """

    result = graph.query(query)
    rows = results_to_list(result.result_set, result.header) if result.result_set else []
    out = _truncate(to_json({
        "source_id": source_id,
        "target_id": target_id,
        "max_path_length": max_path_length,
        "total_paths": len(rows),
        "paths": rows,
    }))
    return [TextContent(type="text", text=out)]


async def _handle_picsure_search(arguments: dict, graph) -> list[TextContent]:
    phv_ids  = arguments.get("phv_ids", [])
    keyword  = arguments.get("keyword", "")
    semantic = arguments.get("semantic", True)
    limit    = arguments.get("limit", 20)

    if not phv_ids and not keyword:
        return [TextContent(type="text", text=to_json({"error": "Provide phv_ids or keyword"}))]

    enrichment_info  = None
    warnings         = []
    original_keyword = keyword
    phv_to_concepts: dict = {}

    # Semantic mode: enrich keyword → KG → phv IDs, then PIC-SURE path lookup
    if keyword and semantic:
        synonyms = await fetch_synonyms(keyword)
        curies   = synonyms["curies"]
        labels   = synonyms["labels"]
        warnings.extend(synonyms.get("warnings", []))
        enrichment_info = {"curies": curies, "labels": labels}

        where_expr = _build_synonym_where(curies, labels, keyword)

        kg_query = f"""
        MATCH (concept)-[r]-(v:`biolink.StudyVariable`)
        WHERE ({where_expr})
          AND NOT labels(concept)[0] = 'biolink.StudyVariable'
        RETURN DISTINCT v.id AS variable_id, concept.id AS concept_id, concept.name AS concept_name
        LIMIT {limit * 5}
        """
        kg_result = graph.query(kg_query)
        kg_rows   = results_to_list(kg_result.result_set, kg_result.header) if kg_result.result_set else []

        kg_phv_ids = []
        for row in kg_rows:
            vid = row.get("variable_id") or ""
            phv = vid.split(".")[0] if "." in vid else vid
            if not phv.startswith("phv"):
                continue
            if phv not in phv_to_concepts:
                phv_to_concepts[phv] = []
                kg_phv_ids.append(phv)
            concept_entry = {"concept_id": row.get("concept_id"), "concept_name": row.get("concept_name")}
            if concept_entry not in phv_to_concepts[phv]:
                phv_to_concepts[phv].append(concept_entry)

        phv_ids = list(dict.fromkeys(list(phv_ids) + kg_phv_ids))
        keyword = ""  # phv IDs now drive the PIC-SURE lookup

    # Strip version suffixes: "phv00425822.v1.p1" → "phv00425822"
    search_terms = [pid.split(".")[0] if "." in pid else pid for pid in phv_ids]
    if keyword:
        search_terms.append(keyword)
    search_terms = list(dict.fromkeys(search_terms))  # deduplicate

    results: list = []
    seen_paths: set = set()
    async with httpx.AsyncClient(timeout=15.0) as client:
        responses = await asyncio.gather(*[
            client.post(PICSURE_SEARCH_URL, json={"query": term})
            for term in search_terms
        ], return_exceptions=True)

    for term, resp in zip(search_terms, responses):
        if len(results) >= limit:
            break
        if isinstance(resp, Exception):
            warnings.append(f"PIC-SURE search failed for '{term}': {resp}")
            continue
        if resp.status_code != 200:
            warnings.append(f"PIC-SURE returned HTTP {resp.status_code} for '{term}'")
            continue
        try:
            phenotypes = resp.json().get("results", {}).get("phenotypes", {})
            for path, meta in phenotypes.items():
                if len(results) >= limit:
                    break
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                parts      = [p for p in path.strip("\\").split("\\") if p]
                study      = parts[0] if parts else None
                phv        = next((p for p in parts if p.startswith("phv")), None)
                vname      = parts[-1] if len(parts) > 1 else None
                cat_values = meta.get("categoryValues", [])
                entry = {
                    "picsure_path": path,
                    "study": study,
                    "phv_id": phv,
                    "variable_name": vname,
                    "categorical": meta.get("categorical"),
                    "category_values": cat_values[:20],
                    "total_category_values": len(cat_values),
                }
                if phv and phv in phv_to_concepts:
                    entry["matched_concepts"] = phv_to_concepts[phv]
                results.append(entry)
        except Exception as e:
            warnings.append(f"Failed to parse PIC-SURE response for '{term}': {e}")

    out = _truncate(to_json({
        "keyword": original_keyword if original_keyword else None,
        "mode": "semantic" if (original_keyword and semantic) else "direct",
        **({"enrichment": enrichment_info} if enrichment_info else {}),
        "total_variables_found": len(results),
        "variables": results,
        **({"warnings": warnings} if warnings else {}),
    }))
    return [TextContent(type="text", text=out)]


async def _handle_find_cohort_variables(arguments: dict, graph) -> list[TextContent]:
    concepts             = arguments.get("concepts", [])
    variables_per_concept = arguments.get("variables_per_concept", 20)
    limit                = arguments.get("limit", 10)

    if not concepts:
        return [TextContent(type="text", text=to_json({"error": "Provide at least one concept"}))]

    # Step 1: Parallel synonym enrichment for all concepts
    enrichments = await asyncio.gather(
        *[fetch_synonyms(c) for c in concepts],
        return_exceptions=True,
    )

    all_warnings = []
    concept_data = []
    for concept, enrichment in zip(concepts, enrichments):
        if isinstance(enrichment, Exception):
            all_warnings.append(f"Enrichment failed for '{concept}': {enrichment}")
            curies, labels = [], [concept]
        else:
            curies = enrichment["curies"]
            labels = enrichment["labels"]
            all_warnings.extend(enrichment.get("warnings", []))
        concept_data.append({"concept": concept, "curies": curies, "labels": labels, "variables": []})

    # Step 2: KG query per concept to get linked StudyVariables
    for cdata in concept_data:
        where_expr = _build_synonym_where(cdata["curies"], cdata["labels"], cdata["concept"])
        kg_query = f"""
        MATCH (concept)-[r]-(v:`biolink.StudyVariable`)
        WHERE ({where_expr})
          AND NOT labels(concept)[0] = 'biolink.StudyVariable'
        RETURN DISTINCT v.id AS variable_id, v.name AS variable_name
        LIMIT {variables_per_concept * 5}
        """
        try:
            kg_result = graph.query(kg_query)
            kg_rows   = results_to_list(kg_result.result_set, kg_result.header) if kg_result.result_set else []
            seen_ids: set = set()
            for row in kg_rows:
                vid = row.get("variable_id")
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    cdata["variables"].append({"variable_id": vid, "variable_name": row.get("variable_name")})
                    if len(cdata["variables"]) >= variables_per_concept:
                        break
        except Exception as e:
            all_warnings.append(f"KG query failed for '{cdata['concept']}': {e}")

    # Step 3: Build variable_id → concepts mapping
    var_to_concepts: dict = {}
    for cdata in concept_data:
        for var in cdata["variables"]:
            vid = var["variable_id"]
            if vid:
                if vid not in var_to_concepts:
                    var_to_concepts[vid] = []
                if cdata["concept"] not in var_to_concepts[vid]:
                    var_to_concepts[vid].append(cdata["concept"])

    all_var_ids = list(var_to_concepts.keys())

    # Step 4: KG query to map variable_ids → studies directly (no PIC-SURE)
    study_concepts: dict  = {}
    study_variables: dict = {}
    if all_var_ids:
        id_list = ", ".join(f'"{vid}"' for vid in all_var_ids)
        kg_study_query = f"""
        MATCH (v:`biolink.StudyVariable`)-[]-(s:`biolink.Study`)
        WHERE v.id IN [{id_list}]
        RETURN v.id AS variable_id, v.name AS variable_name, s.id AS study_id, s.name AS study_name
        """
        try:
            study_result = graph.query(kg_study_query)
            study_rows = results_to_list(study_result.result_set, study_result.header) if study_result.result_set else []
            for row in study_rows:
                vid      = row["variable_id"]
                study_id = row["study_id"]
                if not study_id:
                    continue
                if study_id not in study_concepts:
                    study_concepts[study_id]  = set()
                    study_variables[study_id] = []
                for c in var_to_concepts.get(vid, []):
                    study_concepts[study_id].add(c)
                existing = {v["variable_id"] for v in study_variables[study_id]}
                if vid not in existing:
                    study_variables[study_id].append({
                        "variable_id":      vid,
                        "variable_name":    row.get("variable_name"),
                        "study_id":         study_id,
                        "study_name":       row.get("study_name"),
                        "matched_concepts": var_to_concepts.get(vid, []),
                    })
        except Exception as e:
            all_warnings.append(f"KG study lookup failed: {e}")

    all_concept_names = {cdata["concept"] for cdata in concept_data}
    feasible_studies  = []
    partial_studies   = []
    for study_id, concepts_present in sorted(study_concepts.items(), key=lambda x: len(x[1]), reverse=True):
        missing = all_concept_names - concepts_present
        entry = {
            "study_id":        study_id,
            "study_name":      (study_variables[study_id][0]["study_name"] if study_variables[study_id] else None),
            "concepts_found":  sorted(concepts_present),
            "concepts_missing": sorted(missing),
            "variables":       study_variables[study_id],
        }
        if not missing:
            feasible_studies.append(entry)
        else:
            partial_studies.append(entry)

    feasible_studies = feasible_studies[:limit]
    partial_studies  = partial_studies[:5]

    out = _truncate(to_json({
        "concepts_searched": [cdata["concept"] for cdata in concept_data],
        "enrichment_summary": [
            {
                "concept":        cdata["concept"],
                "curies_found":   len(cdata["curies"]),
                "variables_in_kg": len(cdata["variables"]),
            }
            for cdata in concept_data
        ],
        "feasible_studies_count": len(feasible_studies),
        "feasible_studies": feasible_studies,
        "partial_studies":  partial_studies,
        **({"warnings": all_warnings} if all_warnings else {}),
    }))
    return [TextContent(type="text", text=out)]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "trapi_query":                     _handle_trapi_query,
    "cypher_query":                    _handle_cypher_query,
    "search_concepts":                 _handle_search_concepts,
    "get_concept_graph":               _handle_get_concept_graph,
    "get_concept_connections":         _handle_get_concept_connections,
    "list_graph_schema":               _handle_list_graph_schema,
    "find_highly_connected_variables": _handle_find_highly_connected_variables,
    "search_variables_by_name":        _handle_search_variables_by_name,
    "expand_concept":                  _handle_expand_concept,
    "find_concept_paths":              _handle_find_concept_paths,
    "picsure_search":                  _handle_picsure_search,
    "find_cohort_variables":           _handle_find_cohort_variables,
}


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch incoming tool calls to their dedicated handler functions."""
    try:
        _, graph = get_redis_connection()
        handler  = _TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=to_json({"error": f"Unknown tool: {name}"}))]
        return await handler(arguments, graph)
    except Exception as e:
        return [TextContent(type="text", text=to_json({"error": f"Error executing tool '{name}': {str(e)}"}))]


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def create_sse_app():
    """Create a plain ASGI app that serves the MCP server over SSE."""
    sse = SseServerTransport("/messages/")

    async def asgi_app(scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] == "http":
            if path == "/health":
                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"status":"ok"}'})
            elif path == "/sse":
                async with sse.connect_sse(scope, receive, send) as streams:
                    await app.run(streams[0], streams[1], app.create_initialization_options())
            elif path.startswith("/messages/"):
                await sse.handle_post_message(scope, receive, send)
            else:
                await send({"type": "http.response.start", "status": 404, "headers": []})
                await send({"type": "http.response.body", "body": b"Not found"})
        elif scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await send({"type": "lifespan.shutdown.complete"})

    return asgi_app


async def run_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redis Graph MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="Transport mode (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0", help="SSE host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="SSE port (default: 8000)")
    args = parser.parse_args()

    if args.transport == "sse":
        print(f"Starting Redis Graph MCP server (SSE) on http://{args.host}:{args.port}/sse")
        starlette_app = create_sse_app()
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    else:
        asyncio.run(run_stdio())
