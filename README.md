# RAG with LangChain

A minimal Retrieval-Augmented Generation (RAG) pipeline built with [LangChain](https://www.langchain.com/), [OpenAI](https://platform.openai.com/), and [Pinecone](https://www.pinecone.io/). It retrieves the most relevant chunks from a vector store and uses them as grounded context for an LLM answer.

## Overview

The pipeline is composed end-to-end with [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/):

```
question ──► retriever ──► format docs ──► prompt ──► LLM ──► answer
```

- **Embeddings** — `OpenAIEmbeddings`
- **Vector store** — Pinecone (`PineconeVectorStore`)
- **Retriever** — top-k similarity search (k = 3)
- **LLM** — `ChatOpenAI`
- **Output parser** — `StrOutputParser`

## Requirements

- Python `>=3.14` (see [.python-version](.python-version))
- An OpenAI API key
- A Pinecone account with an index already created
- [`uv`](https://github.com/astral-sh/uv) for dependency management (recommended)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-with-langchain

# Install dependencies with uv
uv sync
```

Or, with `pip`:

```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
INDEX_NAME=your-pinecone-index
```

The Pinecone index must already exist and use a dimension that matches your embedding model (1536 for `text-embedding-3-small` / `text-embedding-ada-002`).

## Usage

Run the entry point:

```bash
uv run main.py
```

To wire the chain into a real query, use the LCEL chain built in [main.py](main.py):

```python
from main import create_retrieval_chain_with_lcel

chain = create_retrieval_chain_with_lcel()
answer = chain.invoke({"question": "What is RAG?"})
print(answer)
```

LCEL gives you streaming, async, and batch out of the box:

```python
# Streaming
for chunk in chain.stream({"question": "Explain hybrid search."}):
    print(chunk, end="", flush=True)

# Async
answer = await chain.ainvoke({"question": "What is HyDE?"})

# Batch
answers = chain.batch([
    {"question": "What is a reranker?"},
    {"question": "What is top-k?"},
])
```

## Project Structure

```
rag-with-langchain/
├── main.py              # LCEL retrieval chain
├── documentation.txt    # Sample knowledge base content
├── pyproject.toml       # Project metadata and dependencies
├── .env                 # Local secrets (not committed)
└── README.md
```

## How It Works

1. The user's question is passed through the retriever, which embeds it and pulls the top-3 similar chunks from Pinecone.
2. `format_docs` joins those chunks into a single context string.
3. The prompt template injects `{context}` and `{question}` and asks the LLM to answer **only** from the provided context.
4. `ChatOpenAI` generates the response, and `StrOutputParser` returns plain text.

See [documentation.txt](documentation.txt) for a deeper write-up on RAG concepts, retrieval strategies, and best practices.

## License

MIT
