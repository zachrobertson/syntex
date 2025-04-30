import numpy as np
import json
from typing import List, Optional
from sqlite3 import Connection
from pydantic import BaseModel, ValidationError
from sqlmodel import Session
from sqlalchemy import text, event

from providers.provider import ModelProvider
from models import SearchResult, SearchQuery, RagResponse

class ModelResponse(BaseModel):
    response: str
    context: List[str]

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
            SELECT document_id, content
            FROM documents
            WHERE id = :rowid
        """)
        doc_result = session.exec(doc_stmt, params={"rowid": rowid}).first()
        if doc_result:
            search_results.append(
                SearchResult(
                    id=doc_result[0],  # document_id
                    content=doc_result[1],  # content
                    similarity=1.0 - float(distance)  # Convert distance to similarity
                )
            )
    
    return search_results

def rag(query: str, session: Connection, model: ModelProvider, max_retries: int = 3) -> RagResponse:
    """RAG Steps:
    1. User providers a query they want to search the document database with (input.query)
    2. This query is turned into an embedding (using the model.embed method)
    3. Do KNN search on the database (using the knn method above)
    4. Convert the KNN responses to text by querying database
    5. Construct query with KNN text response
    6. Return response with references to documents that matched
    """
    # Step 2: Generate embedding for the query
    query_embedding = model.embed(query)
    
    # Step 3/4: Perform KNN search and get content from documents
    search_query = SearchQuery(
        query_embedding=query_embedding,
        top_k=10
    )
    knn_results = search(search_query, session)
    context_ids = [result.id for result in knn_results]
    
    # Step 5: Construct the prompt with context
    context = "\n\n".join([f"ID: {result.id}\n CONTENT:\n {result.content}" for result in knn_results])
    
    # Initialize variables for retry logic
    last_error = None
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Construct the prompt with error information if this is a retry
            error_context = f"\nPrevious error: {last_error}" if last_error else ""
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't know".
Respond with a JSON formatted object with the following schema:
{{
    "response": str,
    "context": List[str]
}}

Do not add any markdown quotes around the JSON object and be careful to escape any quotes or brackets correctly so that the string can be parsed in python using json.loads().
The `context` list should be a list of ids from the context string below that were used to answer the users query. If not context is useful in answering the questions then an
empty array should be returned.
{error_context}

Context:
{context}

Question: {query}

Answer:"""
            
            # Step 6: Generate and validate the response
            response_text = model.genText(prompt)
            try:
                # Try to parse the response as JSON
                response_json = json.loads(response_text)
                # Validate against our Pydantic model
                model_response = ModelResponse(**response_json)
                # If we get here, validation succeeded
                return RagResponse(
                    response=model_response.response,
                    context=model_response.context
                )
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                retry_count += 1
                if retry_count == max_retries:
                    raise ValueError(f"Failed to generate valid JSON response after {max_retries} attempts. Last error: {last_error}")
                continue
                
        except Exception as e:
            raise ValueError(f"Unexpected error during RAG process: {str(e)}")
