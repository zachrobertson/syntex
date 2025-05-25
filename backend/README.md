# Syntex - Server

A REST API for LLM interactions with an advanced RAG functionality. It is written in Python using the FastAPI library.

## Setup

It is suggested that you use a virtual environment
- `python3 -m venv .venv`
- Linux: `source .venv/bin/activate`, Windows: `.\.venv\Scripts\activate`
- Install packages `cd backend && pip install -r requirements.txt`

## Usage

```bash
usage: Syntex CLI interface [-h] [--database DATABASE] [--host HOST] [--port PORT]
                            [--clean]

options:
  -h, --help           show this help message and exit
  --database DATABASE  Path to local SQLite database file
  --host HOST          Hostname for the RAG frontend
  --port PORT          Port for the RAG interface
  --clean              Clean the database before starting the server
```

## Running

- `cd backend && python server.py`

## OpenAPI documentation

You can find the documentation for the API interface by running the server and going to the `/docs` endpoint in the browser.
