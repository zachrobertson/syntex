import uuid
import json
import time
from typing import List, Dict, Any
from sqlite3 import Connection
from pydantic import ValidationError
from sqlmodel import Session
from sqlalchemy import text

from providers.provider import ModelProvider
from models import SearchResult, SearchQuery, ChatMessage, RagResponse
from prompt_utils import create_structured_prompt, parse_and_validate_with_model

def search(
        query: SearchQuery,
        session: Session,
) -> List[SearchResult] :
    # Convert the embedding to a JSON array string
    vec_json = json.dumps(query.query_embedding)
    
    # Perform vector similarity search
    stmt = text("""
        SELECT 
            v.rowid,
            v.distance
        FROM vec_items v
        JOIN documents d on d.id = v.rowid
        WHERE embedding MATCH :query AND k = :k
        ORDER BY distance
    """)
    
    # First get the rowids and distances
    results = session.exec(stmt, params={"query": vec_json, "k": query.top_k}).fetchall()
    
    # Then get the document details for the matching rows
    search_results = []
    for rowid, distance in results:
        doc_stmt = text("""
            SELECT id, filename, path, content
            FROM documents
            WHERE id = :rowid
        """)
        doc_result = session.exec(doc_stmt, params={"rowid": rowid}).first()
        if doc_result:
            search_results.append(
                SearchResult(
                    id=doc_result[0],  # document id
                    filename=doc_result[1], # filename
                    path=doc_result[2], # relative path
                    content=doc_result[3],  # content
                    similarity=1.0 - float(distance)  # Convert distance to similarity
                )
            )
    
    return search_results

async def rag(user_message: ChatMessage, session: Connection, model: ModelProvider, max_retries: int = 3) -> ChatMessage:
    query_embedding = await model.embed(user_message.content)
    search_query = SearchQuery(
        query_embedding=query_embedding,
        top_k=10
    )
    knn_results = search(search_query, session)
    context = "\n\n".join([f"ID: {result.id}\n CONTENT:\n {result.content}" for result in knn_results])
    
    task_description = """
Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't know".

The `context` list should be a list of ids from the context string below that were used to answer the user's query. If no context is useful in answering the question then an empty array should be returned.
"""

    # Define the response schema for the prompt
    response_schema = {
        "response": "str",
        "context": "List[str]"
    }
    
    context_data = {
        "Context": context,
        "Question": user_message.content
    }
    
    last_error = None
    retry_count = 0
    while retry_count < max_retries:
        try:
            prompt = create_structured_prompt(
                task_description=task_description,
                response_schema=response_schema,
                context=context_data,
                last_error=last_error
            )

            response_text = await model.genText(prompt)
            
            # Parse and validate using our Pydantic model
            validated_response, error = parse_and_validate_with_model(
                response_text=response_text,
                model_class=RagResponse
            )
            
            if error:
                last_error = error
                retry_count += 1
                if retry_count == max_retries:
                    raise ValueError(f"Failed to generate valid JSON response after {max_retries} attempts. Last error: {last_error}")
                continue
            
            return ChatMessage(
                id=str(uuid.uuid4()),
                session_id=user_message.session_id,
                role="assistant",
                content=validated_response.response,
                context_ids=validated_response.context if validated_response.context else None,
                created_at=int(time.time())
            )
                
        except Exception as e:
            raise ValueError(f"Unexpected error during RAG process: {str(e)}")
