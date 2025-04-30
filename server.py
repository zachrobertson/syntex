import uvicorn
import argparse
import os
import site
import struct
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile

from dotenv import load_dotenv
from sqlalchemy import text, event
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from sqlmodel import SQLModel, Session, create_engine, select

from rag import search, rag
from providers.openai import OpenAIModel, ModelProvider
from models import Documents, PyRag, SearchQuery, VectorItem, ChatMessage, ChatSession

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
    
    # Set server start time
    app.state.server_start_time = int(time.time())
    
    yield
    # Cleanup
    engine.dispose()

app = FastAPI(lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="templates/static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dependency to get database session
def get_session():
    with Session(engine) as session:
        yield session


def create_vector_table(session: Session):
    """Create the virtual table for vector items"""
    session.exec(text(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{embedding_dim}])"))
    session.commit()


@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

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
            embedding = model.embed(content)
            
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
    
@app.get("/get-document/{doc_id}")
async def get_document(
    doc_id: str,
    session: Session = Depends(get_session)
):
    """
    Get a document's content by its ID
    """
    try:
        # Query the document by document_id
        statement = select(Documents).where(Documents.document_id == doc_id)
        document = session.exec(statement).first()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
        return {
            "id": document.document_id,
            "content": document.content
        }
    except HTTPException:
        raise
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
    List all documents in the database with directory structure
    """
    try:
        statement = select(Documents)
        documents = session.exec(statement).all()
        
        # Organize documents by directory structure
        doc_tree = {}
        for doc in documents:
            path_parts = doc.path.split('/') if doc.path else []
            current_level = doc_tree
            
            # Build the tree structure
            for part in path_parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            
            # Add the document to the current level
            if path_parts:
                current_level[path_parts[-1]] = {
                    "id": doc.document_id,
                    "content": doc.content,
                    "created_at": doc.created_at,
                    "is_file": True
                }
            else:
                # Root level document
                doc_tree[doc.document_id] = {
                    "id": doc.document_id,
                    "content": doc.content,
                    "created_at": doc.created_at,
                    "is_file": True
                }
        
        return {"documents": doc_tree}
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

@app.post("/chat-session")
async def create_chat_session(
    name: str = Body(..., embed=True),
    session: Session = Depends(get_session)
):
    """
    Create a new chat session
    """
    try:
        chat_session = ChatSession(
            name=name,
            created_at=int(time.time()),
            updated_at=int(time.time())
        )
        session.add(chat_session)
        session.commit()
        return {"id": chat_session.id, "name": chat_session.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-sessions")
async def list_chat_sessions(
    session: Session = Depends(get_session)
):
    """
    List all chat sessions
    """
    try:
        statement = select(ChatSession).order_by(ChatSession.updated_at.desc())
        sessions = session.exec(statement).all()
        return {
            "sessions": [
                {
                    "id": sess.id,
                    "name": sess.name,
                    "created_at": sess.created_at,
                    "updated_at": sess.updated_at
                }
                for sess in sessions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-session/{session_id}")
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

        return {
            "session": {
                "id": chat_session.id,
                "name": chat_session.name,
                "created_at": chat_session.created_at,
                "updated_at": chat_session.updated_at
            },
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "context_ids": json.loads(msg.context_ids) if msg.context_ids else None,
                    "created_at": msg.created_at
                }
                for msg in messages
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(
    input: str = Body(..., embed=True),
    session_id: int = Body(..., embed=True),
    session: Session = Depends(get_session),
    model: OpenAIModel = Depends(get_model)
):
    """
    Process a chat message and return a response
    """
    try:
        # Verify session exists
        chat_session = session.get(ChatSession, session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Save user message
        user_message = ChatMessage(
            session_id=session_id,
            role="user",
            content=input,
            created_at=int(time.time())
        )
        session.add(user_message)
        session.commit()

        # Get RAG response
        response = rag(input, session, model)
        
        # Save assistant message
        assistant_message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=response.response,
            context_ids=json.dumps(response.context) if response.context else None,
            created_at=int(time.time())
        )
        session.add(assistant_message)
        
        # Update session timestamp
        chat_session.updated_at = int(time.time())
        session.commit()

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history")
async def get_chat_history(
    session: Session = Depends(get_session)
):
    """
    Get the chat history
    """
    try:
        statement = select(ChatMessage).order_by(ChatMessage.created_at)
        messages = session.exec(statement).all()
        return {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "context_ids": json.loads(msg.context_ids) if msg.context_ids else None,
                    "created_at": msg.created_at
                }
                for msg in messages
            ]
        }
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
        return search(query, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                embedding = model.embed(content)
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
                    
                    # Get the relative path from webkitRelativePath
                    rel_path = file.filename
                    if hasattr(file, 'webkitRelativePath'):
                        rel_path = file.webkitRelativePath
                    
                    # Add document to Documents table
                    db_document = Documents(
                        document_id=file.filename,
                        content=content,
                        path=rel_path
                    )
                    session.add(db_document)
                    session.commit()
                    
                    # Generate embedding
                    embedding = model.embed(content)
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
                    results.append({"status": "error", "message": f"Error processing {file.filename}: {str(e)}"})
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/server-status")
async def get_server_status():
    """
    Get server status including start time
    """
    return {
        "start_time": app.state.server_start_time,
        "clean_mode": app.state.clean_database
    }

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