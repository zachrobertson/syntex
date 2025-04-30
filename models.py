from pydantic import BaseModel
from typing import List, Optional
from sqlmodel import SQLModel, Field

# SQL model for documents table
class Documents(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(unique=True)
    content: str
    path: str | None = Field(default=None)  # Store relative path within directory
    created_at: int | None = Field(default=None, index=True)

# Pydantic CLI model
class PyRag(BaseModel):
    database: str
    host: str
    port: int
    clean: bool = False

# Pydantic models for request/response
class Document(BaseModel):
    id: str
    content: str

class RagResponse(BaseModel):
    response: str
    context: Optional[List[str]] = None

class SearchQuery(BaseModel):
    query_embedding: List[float]
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    content: str
    similarity: float

class VectorItem(BaseModel):
    rowid: int
    document_id: str
    content: str
    vector: List[float]

class ChatSession(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: int | None = Field(default=None, index=True)
    updated_at: int | None = Field(default=None, index=True)

class ChatMessage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id")
    role: str  # 'user' or 'assistant'
    content: str
    context_ids: str | None = None  # JSON string of context IDs
    created_at: int | None = Field(default=None, index=True)