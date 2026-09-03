# DUG MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server for querying a biomedical knowledge graph stored in Redis (RedisGraph). Supports both **stdio** (Claude Desktop) and **SSE** (HTTP) transports.

## Overview

This server exposes a biomedical knowledge graph as MCP tools and prompts, enabling LLMs and AI agents to:

- Search for diseases, phenotypes, and other biomedical concepts by name
- Enrich searches with synonyms using [Name Resolution SRI](https://name-resolution-sri.renci.org) and [SAP-BERT](https://github.com/cambridgeltl/sapbert)
- Discover study variables connected to a concept via **semantic biolink relationships** — not keyword matching
- Look up variable paths in [BioData Catalyst PIC-SURE](https://picsure.biodatacatalyst.nhlbi.nih.gov/) to enable cohort building
- Traverse concept relationships and find paths between entities
- Execute TRAPI and raw Cypher queries for advanced use cases

## How Variable Discovery Works

The key design principle: **the Knowledge Graph decides which variables are relevant. PIC-SURE confirms the data exists.**

```
Natural language query: "asthma"
            │
            ▼
  Step 1 — Synonym Enrichment
    Name Resolution SRI + SAP-BERT
    "asthma" → MONDO:0004979, MONDO:0004784,
               HP:0002099, UMLS:C0004096 ...
            │
            ▼
  Step 2 — KG Semantic Query
    MATCH (concept)-[biolink:related_to]-(variable:StudyVariable)
    WHERE concept.id IN [enriched CURIEs]
    → Only variables with a proven biolink relationship
      are returned. No keyword noise.
            │
            ▼
  Step 3 — PIC-SURE Path Lookup (picsure_search)
    phv IDs → full BDC variable paths
    → Confirms the variable exists in BDC data
            │
            ▼
  Step 4 — Cohort Query (with BDC auth token)
    PIC-SURE /query/sync → participant count
```

This pipeline is fundamentally different from PIC-SURE keyword search: a variable only reaches PIC-SURE if the Knowledge Graph has established a semantic relationship between the concept and that variable via a biolink edge.

## Architecture

```
User / Claude Desktop / API Client
            │
            ▼
    redis-graph-agent  (port 8080)
      - REST API for natural language queries
      - LLM reasoning via vLLM (tool-calling mode)
            │
            ▼
    redis-mcp-server  (port 8000)
      - MCP tools and prompts over SSE
      - Synonym enrichment: Name Resolution SRI + SAP-BERT
      - KG semantic variable discovery
      - PIC-SURE path lookup (open resource, no auth)
            │
            ├──────────────────────┐
            ▼                      ▼
      Redis (RedisGraph)     PIC-SURE API
      Knowledge Graph        BioData Catalyst
```

---

## Using with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "redis-graph": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Restart Claude Desktop after updating the config.

---

## Using the Agent API

### Via Swagger UI

Navigate to `/docs` on the running agent service.

### Via curl

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "find study variables related to asthma"}'
```

### Request body

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural language question |
| `max_results` | int | 20 | Max results per tool call |
| `system_prompt` | string | built-in | Override the system prompt |

### Response

```json
{
  "answer": "Found 20 study variables related to asthma...",
  "tools_used": ["search_concepts"],
  "tool_results": [...]
}
```

---

## Available Tools

All tools return JSON. Below is each tool, what it does, and a concrete example.

---

### `search_concepts`

Search for biomedical concepts by name, or find study variables related to a concept with **automatic synonym enrichment**.

When `find_variables=true`, the term is expanded via Name Resolution SRI and SAP-BERT (up to 10 results each). The combined CURIEs and labels are used to search the graph for connected `StudyVariable` nodes. Results are **deduplicated by variable** — each variable appears once with a `matched_concepts` array showing which concepts matched and via which predicate. The `limit` applies to unique variables, not raw rows. If either enrichment service is unavailable, a `warnings` field is included in the response so partial results are never silent.

**Find concepts by name:**
```json
{ "search_term": "asthma" }
```

**Filter by node type:**
```json
{ "search_term": "asthma", "node_type": "Disease" }
```

**Find variables with synonym enrichment:**
```json
{ "search_term": "asthma", "find_variables": true, "limit": 20 }
```

Example response shape for `find_variables=true`:
```json
{
  "search_term": "asthma",
  "enrichment": { "curies": ["MONDO:0004979", "..."], "labels": ["asthma", "..."] },
  "total_results": 20,
  "variables": [
    {
      "variable_id": "phv00425822.v1.p1",
      "variable_name": "p_asth",
      "variable_description": "lung, asthma/wheezing/reactive airway",
      "matched_concepts": [
        { "concept_id": "MONDO:0004979", "concept_name": "asthma", "concept_type": "biolink:Disease", "predicate": "biolink:related_to" },
        { "concept_id": "MONDO:0004784", "concept_name": "allergic asthma", "concept_type": "biolink:Disease", "predicate": "biolink:related_to" }
      ]
    }
  ]
}
```

---

### `get_concept_graph`

Get the subgraph around a concept.

- **Depth 1** — returns all directly connected nodes of any type
- **Depth 2** — traverses concept → StudyVariable → Study, with a count of how many other concepts each variable links to

**Direct neighbors:**
```json
{ "concept_id": "MONDO:0004979", "expand_depth": 1 }
```

**Variables and studies:**
```json
{ "concept_id": "MONDO:0004979", "expand_depth": 2, "limit": 50 }
```

---

### `get_concept_connections`

List every entity directly connected to a node — relationship type, node type, name, and ID. Returns a type-count summary plus the full leaf-node list. Also works with variable IDs. Optionally filter by node type.

**All connections:**
```json
{ "concept_id": "MONDO:0004979" }
```

**Only connected studies:**
```json
{ "concept_id": "MONDO:0004979", "node_type_filter": "Study" }
```

**Only connected variables:**
```json
{ "concept_id": "MONDO:0004979", "node_type_filter": "StudyVariable" }
```

**Inspect a variable's connections:**
```json
{ "concept_id": "phv00430345.v1.p1" }
```

---

### `search_variables_by_name`

Search for `StudyVariable` nodes by name fragment or by partial ID. Unlike `search_concepts`, this targets only variables and supports ID prefix matching.

**By name fragment:**
```json
{ "search_term": "bmi" }
```

**By partial phv ID:**
```json
{ "search_term": "phv00430" }
```

---

### `find_highly_connected_variables`

Find `StudyVariable` nodes with the most connections to biomedical concepts — useful for discovering the most broadly relevant variables across disease areas.

```json
{ "min_connections": 5, "limit": 20 }
```

---

### `expand_concept`

Traverse the concept graph N hops from a starting node, returning related concepts with hop distance and path relationship types. Optionally filter by relationship type.

**All related concepts within 2 hops:**
```json
{ "concept_id": "MONDO:0004979", "max_hops": 2 }
```

**Filter by relationship type:**
```json
{ "concept_id": "MONDO:0004979", "max_hops": 2, "relationship_types": ["related_to"] }
```

---

### `find_concept_paths`

Find the shortest path(s) between two biomedical concepts.

```json
{
  "source_id": "MONDO:0004979",
  "target_id": "MONDO:0004784",
  "max_path_length": 3
}
```

Returns the node names and relationship types along each path.

---

### `list_graph_schema`

List all node types in the graph with counts. Recommended as a first step for any LLM to understand what data is available.

```json
{ "show_counts": true }
```

---

### `trapi_query`

Query the knowledge graph using a standard [TRAPI (Translator Reasoner API)](https://github.com/NCATSTranslator/ReasonerAPI) query graph. Automatically handles the biolink predicate hierarchy — querying a parent predicate like `biolink:related_to` also matches all child predicates. Returns clean JSON with `category`, `id`, and `name` fields for each node.

**Disease → StudyVariable (by CURIE):**
```json
{
  "qgraph": {
    "nodes": {
      "disease": {"ids": ["MONDO:0004979"], "categories": ["biolink:Disease"]},
      "variable": {"categories": ["biolink:StudyVariable"]}
    },
    "edges": {
      "e0": {"subject": "disease", "object": "variable"}
    }
  },
  "limit": 10
}
```

**PhenotypicFeature → StudyVariable:**
```json
{
  "qgraph": {
    "nodes": {
      "phenotype": {"categories": ["biolink:PhenotypicFeature"]},
      "variable": {"categories": ["biolink:StudyVariable"]}
    },
    "edges": {
      "e0": {"subject": "phenotype", "object": "variable"}
    }
  }
}
```

The `cypher_used` field in the response shows the generated Cypher for transparency and debugging.

---

### `picsure_search`

Look up BioData Catalyst PIC-SURE variable paths by phv ID or keyword. Use this **after** `search_concepts` or `trapi_query` to get the full PIC-SURE paths needed to build a cohort query. No authentication required — uses the open PIC-SURE resource.

**From phv IDs (chain after search_concepts):**
```json
{ "phv_ids": ["phv00425822", "phv00347788"], "limit": 20 }
```

**By keyword (direct lookup):**
```json
{ "keyword": "asthma", "limit": 10 }
```

Example response:
```json
{
  "total_variables_found": 1,
  "variables": [
    {
      "search_term": "phv00425822",
      "picsure_path": "\\phs001514\\pht009816\\phv00425822\\p_asth\\",
      "study": "phs001514",
      "phv_id": "phv00425822",
      "variable_name": "p_asth",
      "categorical": true,
      "category_values": ["Yes"],
      "total_category_values": 1
    }
  ]
}
```

The `picsure_path` is the exact path to use in a PIC-SURE cohort query. Submit it to the PIC-SURE `/query/sync` endpoint with your BDC auth token to get participant counts.

**Recommended chain for cohort building:**
```
search_concepts(find_variables=true) → picsure_search(phv_ids) → PIC-SURE /query/sync
```

---

### `find_cohort_variables`

Find study variables for **multiple biomedical concepts simultaneously** and identify which studies contain variables for all of them — the core tool for multi-condition cohort feasibility analysis.

For each concept, synonym enrichment finds matching variables in the KG. Variables are then looked up in PIC-SURE. Studies are grouped by which concepts they cover: `feasible_studies` have variables for all concepts; `partial_studies` are missing at least one.

A ready-to-submit PIC-SURE query template is generated for the best feasible study.

**Asthma AND obesity cohort:**
```json
{ "concepts": ["asthma", "obesity"], "limit": 10 }
```

**Three-condition query:**
```json
{ "concepts": ["asthma", "obesity", "hypertension"], "variables_per_concept": 30 }
```

Example response shape:
```json
{
  "concepts_searched": ["asthma", "obesity"],
  "enrichment_summary": [
    { "concept": "asthma", "curies_found": 12, "variables_in_kg": 20 },
    { "concept": "obesity", "curies_found": 8, "variables_in_kg": 18 }
  ],
  "feasible_studies_count": 3,
  "feasible_studies": [
    {
      "study": "phs000285",
      "concepts_found": ["asthma", "obesity"],
      "concepts_missing": [],
      "variables": [...]
    }
  ],
  "partial_studies": [...],
  "picsure_query_template": {
    "resourceUUID": "02e23f52-f354-4e8b-992c-d37c8b9ba140",
    "query": {
      "expectedResultType": "COUNT",
      "categoryFilters": { "\\phs000285\\...\\phv00425822\\p_asth\\": ["Yes"] },
      "numericFilters": {}
    },
    "note": "Submit to POST /query/sync with your BDC auth token. Study: phs000285"
  }
}
```

**Recommended chain for multi-condition cohort building:**
```
find_cohort_variables(concepts) → review feasible_studies → submit picsure_query_template to PIC-SURE /query/sync
```

---

### `cypher_query`

Execute a raw Cypher query directly against RedisGraph. Use backticks for biolink labels containing dots.

**Count all diseases:**
```cypher
MATCH (n:`biolink.Disease`) RETURN COUNT(n)
```

**Find concepts connected to a variable:**
```cypher
MATCH (v:`biolink.StudyVariable` {id: "phv00430345.v1.p1"})--(c)
RETURN labels(c)[0] AS type, c.name AS name, c.id AS id
LIMIT 20
```

---

## Available Prompts

When connected via Claude Desktop or another MCP client, these prompts appear as guided workflows:

| Prompt | Arguments | Description |
|---|---|---|
| `find_variables_for_concept` | `concept` | Synonym-enriched search for study variables |
| `explore_concept` | `concept_id` | Full exploration: connections, studies, subgraph |
| `find_studies_for_disease` | `disease_name` | End-to-end: disease name → matching studies |
| `explain_variable` | `variable_id` | Plain-language explanation of what a variable measures |
| `find_path_between_concepts` | `concept_a_id`, `concept_b_id` | Relationship path between two concepts |

---

## Local Development

### Prerequisites

- Python 3.12+
- Redis with RedisGraph module running locally

### Installation

```bash
pip install -r requirements.txt
```

### Run MCP server (stdio mode)

```bash
REDIS_HOST=localhost \
REDIS_PORT=6379 \
REDIS_PASSWORD=<password> \
REDIS_GRAPH_NAME=<graph_name> \
  python3 redismcp_server.py
```

### Run MCP server (SSE mode)

```bash
REDIS_HOST=localhost \
REDIS_PORT=6379 \
REDIS_PASSWORD=<password> \
REDIS_GRAPH_NAME=<graph_name> \
  python3 redismcp_server.py --transport sse --port 8000
```

### Run agent

```bash
MCP_SERVER_URL=http://localhost:8000/sse \
VLLM_URL=<your_vllm_endpoint>/v1 \
MODEL=<model_name> \
  uvicorn agent_api:app --host 0.0.0.0 --port 8080
```

---

## Kubernetes Deployment

```bash
# Deploy all services
kubectl apply -f k8s/mcp-server.yaml
kubectl apply -f k8s/agent.yaml
kubectl apply -f k8s/ingress.yaml

# Restart pods to pull a new image
kubectl rollout restart deployment/redis-mcp-server deployment/redis-graph-agent -n <namespace>

# Check pod status
kubectl get pods -n <namespace> -l 'app in (redis-mcp-server,redis-graph-agent)'
```

---

## Configuration

### MCP Server

| Variable | Description |
|---|---|
| `REDIS_HOST` | Redis hostname |
| `REDIS_PORT` | Redis port |
| `REDIS_PASSWORD` | Redis auth password |
| `REDIS_GRAPH_NAME` | RedisGraph graph name |

### Agent

| Variable | Description |
|---|---|
| `MCP_SERVER_URL` | SSE endpoint of the MCP server (e.g. `http://redis-mcp-server:8000/sse`) |
| `VLLM_URL` | vLLM OpenAI-compatible endpoint |
| `MODEL` | LLM model name served by vLLM |
| `MAX_TOOL_TURNS` | Max tool call iterations per query (default: `5`) |
