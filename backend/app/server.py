import uuid
import uvicorn
import argparse
import os
import site
import struct
import numpy as np
import json
import time

from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import text, event
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, create_engine, select
from typing import List, Dict, Any, Optional, Generic, TypeVar
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Body

from rag import search
from enhanced_rag import enhanced_rag
from providers.openai import OpenAIModel, ModelProvider
from models import Documents, Syntex, SearchQuery, VectorItem, ChatMessage, ChatSession

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
    
    # Initialize model based on provider
    if app.state.provider == "openai":
        app.state.model = OpenAIModel(
            text_model=app.state.text_model,
            embedding_model=app.state.embedding_model
        )
    elif app.state.provider == "replicate":
        from providers.replicate import ReplicateModel
        app.state.model = ReplicateModel(app.state.replicate_config)
    else:
        raise ValueError(f"Unsupported provider: {app.state.provider}")
    
    # Run health check on model provider
    health = await app.state.model.checkHealth()
    if not health:
        raise RuntimeError("Health check failed")
    
    yield
    # Cleanup
    engine.dispose()

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Dependency to get database session
def get_session():
    with Session(engine) as session:
        yield session


def create_vector_table(session: Session):
    """Create the virtual table for vector items"""
    session.exec(text(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{embedding_dim}])"))
    session.commit()


@app.post("/add-document")
async def add_document(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
    model: ModelProvider = Depends(get_model)
):
    """
    Add new documents to the database
    """
    try:
        results = []
        for file in files:
            # Read file content
            content = await file.read()
            content = content.decode('utf-8')
            
            # Add document to Documents table
            db_document = Documents(
                document_id=file.filename,
                content=content
            )
            session.add(db_document)
            session.commit()  # Commit to get the auto-generated ID
            
            # Generate embedding using the model
            embedding = await model.embed(content)
            
            # Convert the embedding to bytes for storage
            embedding_bytes = struct.pack("%sf" % len(embedding), *embedding)
            session.exec(
                text("INSERT INTO vec_items(rowid, embedding) VALUES (:id, :embedding)").bindparams(
                    id=db_document.id,  # Use the auto-generated integer ID
                    embedding=embedding_bytes
                )
            )
            
            session.commit()
            results.append({"status": "success", "message": f"Document {file.filename} added successfully"})
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add-directory")
async def add_directory(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
    model: ModelProvider = Depends(get_model)
):
    """
    Add all valid text files from a directory to the database
    """
    try:
        results = []
        file_count = 0
        max_files = 1000  # Limit total number of files
        
        for file in files:
            if file_count >= max_files:
                results.append({"status": "warning", "message": f"Reached maximum file limit of {max_files}"})
                break
                
            if is_valid_text_file(file.filename):
                try:
                    # Read file content
                    content = await file.read()
                    content = content.decode('utf-8')

                    path = Path(file.filename)

                    # Add document to Documents table
                    db_document = Documents(
                        content=content,
                        filename=path.name,
                        path=str(path.parent),
                        uploaded_at=int(time.time())
                    )
                    session.add(db_document)
                    session.commit()
                    
                    # Generate embedding
                    embedding = await model.embed(content)
                    embedding_bytes = struct.pack("%sf" % len(embedding), *embedding)
                    session.exec(
                        text("INSERT INTO vec_items(rowid, embedding) VALUES (:id, :embedding)").bindparams(
                            id=db_document.id,
                            embedding=embedding_bytes
                        )
                    )
                    
                    session.commit()
                    results.append({"status": "success", "message": f"Added {file.filename}"})
                    file_count += 1
                    
                except Exception as e:
                    raise HTTPException(status_code=409, detail=str(e))
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get-document/{id:path}")
async def get_document(
    id: str,
    session: Session = Depends(get_session)
):
    """
    Get a document's content by its ID
    """
    try:
        # Query the document by document_id
        statement = select(Documents).where(Documents.id == id)
        document = session.exec(statement).first()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found, id: {id}")
            
        return ApiResponse(data=document)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-document/{id}")
async def remove_document(
    id: str,
    session: Session = Depends(get_session)
):
    """
    Remove a document from the database
    """
    try:
        document = session.get(Documents, id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found, id: {id}")
        session.delete(document)
        session.commit()
        return {"status": "success", "message": f"Document removed successfully, id: {id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    data: T
    error: Optional[str] = None
    meta: Optional[dict] = None

@app.get("/list-documents", response_model=ApiResponse[List[Documents]])
async def list_documents(
    session: Session = Depends(get_session)
):
    """
    List all documents in the database with directory structure
    """
    try:
        statement = select(Documents)
        documents = session.exec(statement).all()
        return ApiResponse(data=documents)
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
        
        return ApiResponse(data={
            "vectors": vectors
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-session", response_model=ApiResponse[Dict[str, Any]])
async def create_chat_session(
    name: str = Body(..., embed=True),
    session: Session = Depends(get_session)
):
    """
    Create a new chat session
    """
    try:
        created_time = int(time.time())
        chat_session = ChatSession(
            name=name,
            created_at=created_time,
            updated_at=created_time
        )
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        
        return ApiResponse(data={
            "id": chat_session.id,
            "name": chat_session.name,
            "created_at": chat_session.created_at,
            "updated_at": chat_session.updated_at
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-sessions", response_model=ApiResponse[List[ChatSession]])
async def list_chat_sessions(
    session: Session = Depends(get_session)
):
    """
    List all chat sessions
    """
    try:
        statement = select(ChatSession).order_by(ChatSession.updated_at.desc())
        sessions = session.exec(statement).all()
        return ApiResponse(data=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-session/{session_id}", response_model=ApiResponse[Dict[str, Any]])
async def get_chat_session(
    session_id: int,
    db_session: Session = Depends(get_session)
):
    """
    Get a specific chat session with its messages
    """
    try:
        # Get the session
        chat_session = db_session.get(ChatSession, session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Get messages for this session
        statement = select(ChatMessage).where(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at)
        messages = db_session.exec(statement).all()

        return ApiResponse(data={
            "session": chat_session,
            "messages": messages
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/chat-session/{session_id}/rename", response_model=ApiResponse[ChatSession])
async def rename_chat_session(
    session_id: int,
    new_name: str = Body(..., embed=True),
    session: Session = Depends(get_session)
):
    """
    Rename a chat session
    """
    try:
        # Get the session
        chat_session = session.get(ChatSession, session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Update the name
        chat_session.name = new_name
        chat_session.updated_at = int(time.time())
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)

        return ApiResponse(data=chat_session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ApiResponse[ChatMessage])
async def chat(
    input: str = Body(..., embed=True),
    session_id: int = Body(..., embed=True),
    session: Session = Depends(get_session),
    model: ModelProvider = Depends(get_model)
):
    """
    Process a chat message and return a response
    """
    # Verify session exists
    chat_session = session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    try:
        # Save user message
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=input,
            created_at=int(time.time())
        )
        session.add(user_message)
        session.commit()

        # Get chat history for context
        statement = select(ChatMessage).where(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at)
        chat_history = session.exec(statement).all()

        # Get documents from context
        statement = select(Documents).order_by(Documents.uploaded_at)
        documents = session.exec(statement).all()

        # Use enhanced RAG instead of regular RAG
        response = await enhanced_rag(user_message, chat_history, documents, session, model)
        session.add(response)
        
        # Update session timestamp
        chat_session.updated_at = int(time.time())
        session.commit()

        return ApiResponse(data={
            "id": response.id,
            "session_id": response.session_id,
            "role": response.role,
            "content": response.content,
            "context_ids": format_context_with_metadata(response.context_ids, session),
            "created_at": response.created_at
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history", response_model=ApiResponse[Dict[str, Any]])
async def get_chat_history(
    id: str,
    session: Session = Depends(get_session)
):
    """
    Get the chat history
    """
    try:
        statement = select(ChatMessage).where(ChatMessage.session_id == id).order_by(ChatMessage.created_at)
        messages = session.exec(statement).all()
        return ApiResponse(data={
            "messages": messages
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=ApiResponse[Any])
async def search_documents(
    query: SearchQuery,
    session: Session = Depends(get_session)
):
    """
    Search for similar documents using vector similarity
    """
    try:
        result = search(query, session)
        return ApiResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/embed", response_model=ApiResponse[Any])
async def embed_document(
    query: str,
    model: ModelProvider = Depends(get_model)
):
    """
    Embed a user query using the model provider
    """
    try:
        result = await model.embed(query)
        return ApiResponse(data={
            "query_embedding": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_context_with_metadata(context_ids_str: str, session: Session) -> str:
    """
    Convert context IDs string to a list of document references with metadata
    """
    if not context_ids_str:
        return None
        
    try:
        # Parse the JSON string to get the list of document IDs
        context_ids = json.loads(context_ids_str)
        doc_references = []
        
        # Fetch document metadata for each ID
        for doc_id in context_ids:
            doc = session.get(Documents, doc_id)
            if doc:
                doc_references.append({
                    "id": doc_id,
                    "filename": doc.filename or "Untitled",
                    "path": doc.path or ""
                })
            else:
                # If document not found, just include the ID
                doc_references.append({
                    "id": doc_id,
                    "filename": f"Document {doc_id}",
                    "path": ""
                })
                
        return json.dumps(doc_references)
    except Exception as e:
        print(f"Error formatting context with metadata: {str(e)}")
        return context_ids_str  # Return original string in case of error

def is_valid_text_file(file_path: str) -> bool:
    """Check if a file is a valid text file based on extension."""
    valid_extensions = {'.txt', '.py', '.js', '.html', '.css', '.md', '.json', 
                       '.xml', '.yaml', '.yml', '.java', '.cpp', '.c', '.h', 
                       '.hpp', '.cs', '.go', '.rb', '.php', '.ts', '.tsx', '.jsx'}
    return Path(file_path).suffix.lower() in valid_extensions

async def process_directory(directory: Path, base_path: Path, session: Session, model: ModelProvider) -> List[Dict[str, Any]]:
    """Process a directory recursively and add all valid text files to the database."""
    results = []
    file_count = 0
    max_files = 1000  # Limit total number of files
    
    for item in directory.rglob('*'):
        if file_count >= max_files:
            results.append({"status": "warning", "message": f"Reached maximum file limit of {max_files}"})
            break
            
        if item.is_file() and is_valid_text_file(str(item)):
            try:
                # Calculate relative path
                rel_path = str(item.relative_to(base_path))
                
                # Read file content
                with open(item, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add document to Documents table
                db_document = Documents(
                    document_id=str(item.name),
                    content=content,
                    path=rel_path
                )
                session.add(db_document)
                session.commit()
                
                # Generate embedding
                embedding = await model.embed(content)
                embedding_bytes = struct.pack("%sf" % len(embedding), *embedding)
                session.exec(
                    text("INSERT INTO vec_items(rowid, embedding) VALUES (:id, :embedding)").bindparams(
                        id=db_document.id,
                        embedding=embedding_bytes
                    )
                )
                
                session.commit()
                results.append({"status": "success", "message": f"Added {rel_path}"})
                file_count += 1
                
            except Exception as e:
                results.append({"status": "error", "message": f"Error processing {item}: {str(e)}"})
    
    return results

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Syntex API",
        version="1.0.0",
        description="API for Syntex application",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

class SyntexArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_requirements = {
            'replicate': ['replicate_config'],
            'openai': ['text_model', 'embedding_model']
        }

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        
        # Check provider-specific requirements
        if args.provider in self.provider_requirements:
            for required_arg in self.provider_requirements[args.provider]:
                if not getattr(args, required_arg, None):
                    self.error(f"--{required_arg.replace('_', '-')} is required when using the {args.provider} provider")
        
        return args

def main(args: Syntex):
    # Load environment variables
    load_dotenv()
    
    # Store configuration in app state
    app.state.database_path = args.database
    app.state.clean_database = args.clean
    app.state.provider = args.provider
    app.state.text_model = args.text_model
    app.state.embedding_model = args.embedding_model
    app.state.replicate_config = args.replicate_config
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    parser = SyntexArgumentParser("Syntex CLI interface")
    
    # Base arguments
    base_group = parser.add_argument_group("Base Configuration")
    base_group.add_argument(
        "--database",
        type=str,
        required=False,
        default="data/rag.db",
        help="Path to local SQLite database file"
    )
    base_group.add_argument(
        "--host",
        type=str,
        required=False,
        default="localhost",
        help="Hostname for the RAG frontend"
    )
    base_group.add_argument(
        "--port",
        type=int,
        required=False,
        default=8080,
        help="Port for the RAG interface"
    )
    base_group.add_argument(
        "--clean",
        action="store_true",
        help="Clean the database before starting the server"
    )
    
    # Provider configuration
    provider_group = parser.add_argument_group("Model Provider Configuration")
    provider_group.add_argument(
        "--provider",
        type=str,
        required=False,
        default="openai",
        choices=["openai", "replicate"],
        help="Model provider to use"
    )
    
    # OpenAI specific arguments
    openai_group = parser.add_argument_group("OpenAI Configuration")
    openai_group.add_argument(
        "--text-model",
        type=str,
        required=False,
        default="gpt-4o",
        help="Text generation model to use (for OpenAI provider)"
    )
    openai_group.add_argument(
        "--embedding-model",
        type=str,
        required=False,
        default="text-embedding-3-small",
        help="Embedding model to use (for OpenAI provider)"
    )
    
    # Replicate specific arguments
    replicate_group = parser.add_argument_group("Replicate Configuration")
    replicate_group.add_argument(
        "--replicate-config",
        type=str,
        required=False,
        help="Path to Replicate provider configuration file (required for Replicate provider)"
    )
    
    args = parser.parse_args()
    main(args)