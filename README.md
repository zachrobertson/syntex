# Syntex

A simple chat interface for document understanding that can be run completely locally, both frontend and backend.

## Architecture

Syntex uses a simple React/TypeScript frontend that communicates with a Python backend. The backend allows for the underlying LLM to be run locally or hosted by a cloud LLM provider (currently OpenAI). This lets users do advanced search and questioning on large document repositories without sending propietary data to third parties or avoid paying subscription fees for chat interface tools like ChatGPT and Claude and instead use the third party APIs for LLM generation.

## Environment

- Python >= 3.11.2
- Node.js >= v22.13.1
- Sqlite >= 3.40.1

To use third party APIs you need the following environment variables set (in `.env`)
- `OPENAI_API_KEY` -> For the OpenAI model provider
- `REPLICATE_API_KEY` -> For the Replicate model provider with models hosted by Replicate