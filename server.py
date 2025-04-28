import uvicorn
import argparse
import os
import site
import struct
import numpy as np

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import text, event
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Field, Session, create_engine, select

from providers.openai import OpenAIModel

# Create SQLModel engine
engine = None

# Embedding Dimension
embedding_dim = 1536

# Dependency to get model instance
def get_model():
    return app.state.model

def load_extension(dbapi_connection, connection_record):
    # Get the path to the sqlite_vec extension
    site_packages = site.getsitepackages()[0]
    vec_path = os.path.join(site_packages, 'sqlite_vec', 'vec0.so')
    
    dbapi_connection.enable_load_extension(True)
    dbapi_connection.load_extension(vec_path)
    dbapi_connection.enable_load_extension(False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database connection
    global engine
    
    # If clean flag is set, remove the database file if it exists
    if app.state.clean_database and os.path.exists(app.state.database_path):
        try:
            os.remove(app.state.database_path)
        except Exception as e:
            print(f"Warning: Could not remove database file: {e}")
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(app.state.database_path), exist_ok=True)
    
    engine = create_engine(
        f"sqlite:///{app.state.database_path}",
        connect_args={"check_same_thread": False}
    )
    
    # Register the extension loader
    event.listen(engine, 'connect', load_extension)
    
    # Create tables with new schema
    SQLModel.metadata.create_all(engine)
    
    # Create vector table
    with Session(engine) as session:
        create_vector_table(session)
    
    # Initialize model
    app.state.model = OpenAIModel(
        text_model="gpt-4o",
        embedding_model="text-embedding-3-small"
    )
    
    yield
    # Cleanup
    engine.dispose()

app = FastAPI(lifespan=lifespan)

# Dependency to get database session
def get_session():
    with Session(engine) as session:
        yield session

# SQL model for documents table
class Documents(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(unique=True)
    content: str
    created_at: int | None = Field(default=None, index=True)

def create_vector_table(session: Session):
    """Create the virtual table for vector items"""
    session.exec(text(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{embedding_dim}])"))
    session.commit()

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

class ChatMessage(BaseModel):
    message: str
    context: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

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

@app.post("/add-document")
async def add_document(
    document: Document,
    session: Session = Depends(get_session),
    model: OpenAIModel = Depends(get_model)
):
    """
    Add a new document to the database
    """
    try:
        # Add document to Documents table
        db_document = Documents(
            document_id=document.id,
            content=document.content
        )
        session.add(db_document)
        session.commit()  # Commit to get the auto-generated ID
        
        # Generate embedding using the model
        embedding = model.embed(document.content)
        
        # Convert the embedding to bytes for storage
        embedding_bytes = struct.pack("%sf" % len(embedding), *embedding)
        session.exec(
            text("INSERT INTO vec_items(rowid, embedding) VALUES (:id, :embedding)").bindparams(
                id=db_document.id,  # Use the auto-generated integer ID
                embedding=embedding_bytes
            )
        )
        
        session.commit()
        return {"status": "success", "message": f"Document {document.id} added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-document/{doc_id}")
async def remove_document(
    doc_id: str,
    session: Session = Depends(get_session)
):
    """
    Remove a document from the database
    """
    try:
        document = session.get(Documents, doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        session.delete(document)
        session.commit()
        return {"status": "success", "message": f"Document {doc_id} removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-documents")
async def list_documents(
    session: Session = Depends(get_session)
):
    """
    List all documents in the database
    """
    try:
        statement = select(Documents)
        documents = session.exec(statement).all()
        return {"documents": [
            {
                "id": doc.document_id,
                "content": doc.content,
                "created_at": doc.created_at
            }
            for doc in documents
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-vectors")
async def list_vectors(
    session: Session = Depends(get_session)
):
    """
    List all vectors in the vec_items table
    """
    try:
        results = session.exec(
            text("""
                SELECT v.rowid, v.embedding, d.document_id, d.content 
                FROM vec_items v
                JOIN documents d ON d.id = v.rowid
            """)
        ).fetchall()
        
        vectors = []
        for row in results:
            # Convert binary data back to numpy array
            vector_bytes = row[1]
            vector_array = np.frombuffer(vector_bytes, dtype=np.float32)
            vectors.append(
                VectorItem(
                    rowid=row[0],
                    document_id=row[2],
                    content=row[3],
                    vector=vector_array.tolist()  # Convert numpy array to Python list
                )
            )
        
        return vectors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(
    message: ChatMessage,
    session: Session = Depends(get_session),
    model: OpenAIModel = Depends(get_model)
):
    """
    Process a chat message and return a response
    """
    try:
        # Generate response using the model
        response = model.genText(message.message)
        return ChatResponse(
            response=response,
            sources=message.context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(
    query: SearchQuery,
    session: Session = Depends(get_session)
):
    """
    Search for similar documents using vector similarity
    """
    try:
        # Serialize query embedding
        query_bytes = struct.pack("%sf" % len(query.query_embedding), *query.query_embedding)
        
        # Perform vector similarity search
        stmt = text("""
            SELECT 
                d.document_id,
                d.content,
                v.distance
            FROM vec_items v
            JOIN documents d ON d.id = v.rowid
            WHERE v.embedding MATCH :query
            ORDER BY v.distance
            LIMIT :k
        """)
        
        results = session.exec(stmt, params={"query": query_bytes, "k": query.top_k}).fetchall()
        
        return [
            SearchResult(
                id=str(row[0]),
                content=row[1],
                similarity=1.0 - float(row[2])  # Convert distance to similarity
            )
            for row in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main(args: PyRag):
    # Load environment variables
    load_dotenv()
    # Store database path and clean flag in app state
    app.state.database_path = args.database
    app.state.clean_database = args.clean
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Py-RAG CLI interface")
    parser.add_argument(
        "--database",
        type=str,
        required=False,
        default="data/rag.db",
        help="Path to local SQLite database file"
    )
    parser.add_argument(
        "--host",
        type=str,
        required=False,
        default="localhost",
        help="Hostname for the RAG frontend"
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=8080,
        help="Port for the RAG interface"
    )
    parser.add_argument(
        "--model",
        choices=[
            "gpt-4o",
            "ollama",
            "claude-3.7",
            "claude-3.5"
        ],
        required=False,
        default="ollama"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the database before starting the server"
    )
    args = parser.parse_args()
    main(args)