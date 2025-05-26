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

## Replicate Model Configurations

The `models.replicate.json` file defines the configuration for text generation and embedding models used by the system. This file should contain two main sections:

1. `text`: Configuration for text generation models
2. `embedding`: Configuration for embedding models

Each section must include the following fields:
- `model`: The model identifier on Replicate (e.g., "google-deepmind/gemma-2b-it")
- `version`: The specific version hash of the model
- `input`: The input schema for the model
- `output`: The output schema for the model
- `healthCheck`: A simple input configuration to verify the model is working

To create this file:

1. Go to the model's page on Replicate (e.g., https://replicate.com/google-deepmind/gemma-2b-it)
2. Under the "API" section, find the "Schema" subsection
3. Copy the input and output schemas provided
4. Add the model identifier and version hash
5. Create a simple healthCheck configuration with minimal required inputs

Example structure:
```json
{
    "text": {
        "model": "model-identifier",
        "version": "version-hash",
        "input": {
            // Schema from Replicate API
        },
        "output": {
            // Schema from Replicate API
        },
        "healthCheck": {
            "input": {
                // Minimal required inputs
            }
        }
    },
    "embedding": {
        // Similar structure for embedding model
    }
}
```

The healthCheck configuration should use the minimum required inputs to verify the model is functioning correctly. For text models, this is typically a simple prompt, and for embedding models, it's usually a basic text input.


