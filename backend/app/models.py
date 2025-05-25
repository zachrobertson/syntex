from pydantic import BaseModel
from sqlalchemy import Column, TEXT
from sqlmodel import SQLModel, Field
from typing import List, Optional, Dict, Any

# SQL models for tables and entries
class Documents(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    path: str | None = None
    filename: str | None = None
    content: str | None = Field(sa_column=Column(TEXT))
    uploaded_at: int | None = Field(default=None, index=True)

class ChatSession(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: int | None = Field(default=None, index=True)
    updated_at: int | None = Field(default=None, index=True)

class ChatMessage(SQLModel, table=True):
    id: str | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id")
    role: str  # 'user' or 'assistant'
    content: str
    context_ids: str | None = None  # Array of documents ids
    response_metadata: str | None = None  # JSON string for additional metadata
    created_at: int | None = Field(default=None, index=True)

# Pydantic models for requests, responses and the CLI interface
class Syntex(BaseModel):
    database: str
    host: str
    port: int
    clean: bool = False

class SearchQuery(BaseModel):
    query_embedding: List[float]
    top_k: int = 5

class SearchResult(BaseModel):
    id: int
    filename: str | None = None
    path: str | None = None
    content: str | None = None
    similarity: float | None = None

class VectorItem(BaseModel):
    rowid: int
    document_id: str
    content: str
    vector: List[float]

class RagResponse(BaseModel):
    response: str = Field(..., description="The text response to the user's query")
    context: List[str] = Field(default_factory=list, description="List of document IDs that were used to answer the query")

class RagNecessityResponse(BaseModel):
    needs_rag: bool = Field(..., description="Whether RAG is needed to answer the query")
    reasoning: str = Field(..., description="Brief explanation for the decision")
    reformulated_query: Optional[str] = Field(None, description="Optimized query for search (if needs_rag is true)")
    response: Optional[str] = Field(None, description="Direct response to the user (if needs_rag is false)")

class ReplicateModelInput(BaseModel):
    type: str
    title: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class ReplicateModelOutput(BaseModel):
    type: str
    title: str
    items: Dict[str, Any] = Field(default_factory=dict)

class ReplicateModelSection(BaseModel):
    model: str
    version: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    healthCheck: Dict[str, Dict[str, Any] | str] = Field(..., description="Health check configuration for the model")

class ReplicateModelConfig(BaseModel):
    text: ReplicateModelSection
    embedding: ReplicateModelSection