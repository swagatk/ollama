# Local RAG Demo (Linux)
This demonstrates how we can use large language model (llama3) running locally to chat with a document.

## Prerequisites
- Ubuntu 24.04 LTS
- Python 3.12+
- [Ollama](https://github.com/ollama/ollama) running locally (needed for embeddings even in cloud mode)
- PDF document saved as `paper.pdf` in this directory

## Setup

```bash
python -m venv ~/.virtualenvs/langchainenv
source ~/.virtualenvs/langchainenv/bin/activate
pip install --upgrade pip
pip install langchain langchain-community langchain-core langchain-text-splitters \
           langchain-ollama chromadb pypdf
ollama pull llama3:8b
ollama pull mxbai-embed-large
```

## Run (local models)

```bash
source ~/.virtualenvs/langchainenv/bin/activate
cd /home/swagat/GIT/ollama/local_rag
python rag_demo.py
```

## Run (cloud LLM + local embeddings)

```bash
source ~/.virtualenvs/langchainenv/bin/activate
cd /home/swagat/GIT/ollama/local_rag
export OLLAMA_API_KEY="your-cloud-token"
python rag_demo_cloud.py
```

Type questions at the prompt. Use `exit`, `quit`, or `Ctrl+C` to stop.