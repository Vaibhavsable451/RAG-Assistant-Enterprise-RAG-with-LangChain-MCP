# RAG Assistant — Groq + LangChain + MCP (2026 architecture)

An AI Engineer reference project: a Retrieval-Augmented Generation app with
a governance layer, MCP exposure, and CI/CD deployment to Azure App Service.

## Stack

| Layer            | Choice                                              |
|-------------------|------------------------------------------------------|
| LLM               | Groq API (`langchain-groq`, e.g. `llama-3.3-70b-versatile`) |
| Orchestration     | LangChain                                            |
| Embeddings        | `sentence-transformers` (local, no external embed API) |
| Vector store      | FAISS, local file-based index (**no SQL/DB server**) |
| Frontend          | Streamlit                                            |
| Governance        | Presidio (PII redaction) + moderation + JSONL audit log |
| Agent interop     | MCP server exposing `rag_query` as a callable tool   |
| CI/CD             | GitHub Actions → Azure App Service (**no Kubernetes**) |

## Architecture

```
                          ┌─────────────────────┐
                          │   Streamlit UI       │
                          │   (app.py)            │
                          └──────────┬───────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │      Governance Layer            │
                     │  (PII redaction, moderation,     │
                     │   audit logging)                  │
                     └───────────────┬────────────────┘
                                     │
              ┌──────────────────────┼───────────────────────┐
              │                      │                        │
   ┌──────────▼─────────┐  ┌─────────▼─────────┐   ┌──────────▼─────────┐
   │  Chunking            │  │  Embeddings         │   │  Vector Store (FAISS) │
   │  (rag/chunking.py)   │  │  (rag/embeddings.py)│   │  (rag/vector_store.py)│
   └──────────┬─────────┘  └─────────┬─────────┘   └──────────┬─────────┘
              │                      │                        │
              └──────────────────────┴────────────┬───────────┘
                                                    │
                                         ┌──────────▼─────────┐
                                         │  Retriever            │
                                         └──────────┬─────────┘
                                                    │
                                         ┌──────────▼─────────┐
                                         │  Groq LLM Generator  │
                                         └─────────────────────┘

   MCP Server (mcp/mcp_server.py) exposes the same pipeline as a
   `rag_query` tool for external agents/clients.
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your GROQ_API_KEY
```

## Run locally

```bash
streamlit run app.py
```

## Run the MCP server

```bash
python -m mcp.mcp_server
```

## Run tests

```bash
pytest tests/ -v
```

## Deploy to Azure

1. Create an Azure App Service (Linux, Python 3.11 runtime).
2. Set the Startup Command to `bash startup.sh`.
3. In App Service → Configuration → Application settings, add `GROQ_API_KEY`
   and any other values from `.env.example`.
4. In your GitHub repo, add:
   - Secret `AZURE_CREDENTIALS` (a service-principal JSON from
     `az ad sp create-for-rbac --sdk-auth`).
   - Variable `AZURE_WEBAPP_NAME` with your App Service name.
5. Push to `main` — `.github/workflows/azure-deploy.yml` builds, tests, and deploys.

## Notes on scope

- **No SQL database** — the vector index is a local FAISS file persisted to
  `data/vector_store/`. Swap in a managed vector DB later if you need
  multi-instance scaling.
- **No Kubernetes** — deployment target is Azure App Service (PaaS), which is
  simpler to operate for a single-container Streamlit app. Move to AKS only
  if you need pod-level autoscaling or a multi-service mesh.
