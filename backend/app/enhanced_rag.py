import uuid
import json
import time
import logging
from typing import List, Dict, Any
from sqlite3 import Connection

from providers.provider import ModelProvider
from models import SearchQuery, ChatMessage, Documents, RagResponse, RagNecessityResponse
from rag import search
from prompt_utils import create_structured_prompt, parse_and_validate_with_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhanced_rag")

class EnhancedRagResponse(dict):
    needs_rag: bool
    reformulated_query: str
    response: str
    context_ids: List[str]

async def enhanced_rag(
    user_message: ChatMessage,
    chat_history: List[ChatMessage],
    documents: List[Documents],
    session: Connection,
    model: ModelProvider,
    max_retries: int = 3
) -> ChatMessage:
    """
    Enhanced RAG process that uses LLM to determine if RAG is necessary
    and reformulates queries when needed.
    
    Args:
        user_message: The user's message
        chat_history: Previous messages in the chat session
        session: Database session
        model: The LLM model to use
        max_retries: Maximum number of retries for LLM calls
    
    Returns:
        ChatMessage: The assistant's response
    """
    # Initialize metadata for tracking
    metadata = {
        "process_start_time": time.time(),
        "original_query": user_message.content,
        "response_type": "unknown",
        "timing": {}
    }
    
    try:
        # Format chat history for the LLM
        formatted_history = format_chat_history(chat_history)

        # Format document tree
        document_tree = format_document_tree(documents)
        
        # First, ask the LLM if RAG is necessary and to reformulate the query if needed
        logger.info(f"Determining RAG necessity for query: {user_message.content[:50]}...")
        decision_start = time.time()
        decision = await decide_rag_necessity(user_message.content, formatted_history, document_tree, model)
        metadata["timing"]["decision_time"] = time.time() - decision_start
        metadata["needs_rag"] = decision["needs_rag"]
        metadata["reasoning"] = decision.get("reasoning", "Not provided")
        
        logger.info(f"RAG decision: {decision['needs_rag']}, Reasoning: {decision.get('reasoning', 'Not provided')}")

        # If RAG is not needed, respond directly with the LLM
        if not decision["needs_rag"]:
            logger.info("Using direct LLM response")
            metadata["response_type"] = "direct_llm"
            return ChatMessage(
                id=str(uuid.uuid4()),
                session_id=user_message.session_id,
                role="assistant",
                content=decision["response"],
                context_ids=None,
                response_metadata=json.dumps(metadata),
                created_at=int(time.time())
            )
        
        # If RAG is needed, use the reformulated query for search
        reformulated_query = decision["reformulated_query"]
        metadata["reformulated_query"] = reformulated_query
        logger.info(f"Using reformulated query: {reformulated_query}")
        
        # Generate embeddings for the reformulated query
        try:
            embed_start = time.time()
            query_embedding = await model.embed(reformulated_query)
            metadata["timing"]["embedding_time"] = time.time() - embed_start
            
            # Perform vector search
            search_start = time.time()
            search_query = SearchQuery(
                query_embedding=query_embedding,
                top_k=10
            )
            search_results = search(search_query, session)
            metadata["timing"]["search_time"] = time.time() - search_start
            metadata["search_results_count"] = len(search_results)
            logger.info(f"Found {len(search_results)} search results")
            
            # If no search results were found, fall back to direct LLM response
            if not search_results:
                logger.warning("No search results found, falling back to direct response")
                metadata["response_type"] = "fallback_no_results"
                return ChatMessage(
                    id=str(uuid.uuid4()),
                    session_id=user_message.session_id,
                    role="assistant",
                    content=decision.get("response") or "I don't have specific information about that in my knowledge base.",
                    context_ids=None,
                    response_metadata=json.dumps(metadata),
                    created_at=int(time.time())
                )
            
            # Get context IDs from search results
            context_ids = [str(result.id) for result in search_results]
            
            # Format context for the LLM
            context = "\n\n".join([f"ID: {result.id}\nCONTENT:\n{result.content}" for result in search_results])
            
            # Generate the response using the context, original query, and chat history
            logger.info("Generating response with context")
            generation_start = time.time()
            response_with_context = await generate_response_with_context(
                original_query=user_message.content,
                reformulated_query=reformulated_query,
                context=context,
                chat_history=formatted_history,
                model=model
            )
            metadata["timing"]["generation_time"] = time.time() - generation_start
            metadata["response_type"] = "rag"
            metadata["used_context_count"] = len(response_with_context.get("context", []))
            
            # Calculate total processing time
            metadata["total_processing_time"] = time.time() - metadata["process_start_time"]
            
            # Create and return the response
            return ChatMessage(
                id=str(uuid.uuid4()),
                session_id=user_message.session_id,
                role="assistant",
                content=response_with_context["response"],
                context_ids=json.dumps(response_with_context["context"]),
                response_metadata=json.dumps(metadata),
                created_at=int(time.time())
            )
            
        except Exception as e:
            logger.error(f"Error during RAG processing: {str(e)}")
            # Fallback to direct LLM response if available
            if decision.get("response"):
                logger.info("Falling back to direct LLM response due to error")
                metadata["response_type"] = "fallback_error"
                metadata["error"] = str(e)
                return ChatMessage(
                    id=str(uuid.uuid4()),
                    session_id=user_message.session_id,
                    role="assistant",
                    content=decision["response"],
                    context_ids=None,
                    response_metadata=json.dumps(metadata),
                    created_at=int(time.time())
                )
            else:
                # Ultimate fallback
                raise
    
    except Exception as e:
        logger.error(f"Critical error in enhanced_rag: {str(e)}")
        # Create a generic error response
        metadata["response_type"] = "error"
        metadata["error"] = str(e)
        return ChatMessage(
            id=str(uuid.uuid4()),
            session_id=user_message.session_id,
            role="assistant",
            content="I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
            context_ids=None,
            response_metadata=json.dumps(metadata),
            created_at=int(time.time())
        )

def format_chat_history(chat_history: List[ChatMessage]) -> str:
    """
    Format chat history into a string for the LLM.
    
    Args:
        chat_history: List of previous chat messages
    
    Returns:
        str: Formatted chat history
    """
    if not chat_history:
        return ""
    
    formatted = []
    for msg in chat_history[-6:]:  # Limit to the last 6 messages
        formatted.append(f"{msg.role.capitalize()}: {msg.content}")
    
    return "\n".join(formatted)

def format_document_tree(documents: List[Documents]) -> object:
    """
    Format a list of Documents into a hierarchical tree structure that can be
    stringified and injected into an LLM prompt.
    
    Args:
        documents: List of Documents objects from the database
        
    Returns:
        dict: A structured representation of the documents organized by path
    """
    if not documents:
        return {"documents": [], "count": 0}
    
    # Create a hierarchical structure based on document paths
    doc_tree = {
        "documents": [],
        "count": len(documents)
    }
    
    for doc in documents:
        doc_info = {
            "id": doc.id,
            "path": doc.path,
            "filename": doc.filename,
            "preview": doc.content[:100] + "..." if doc.content else None,
            "uploaded_at": doc.uploaded_at
        }
        doc_tree["documents"].append(doc_info)
    
    return doc_tree

async def decide_rag_necessity(
    query: str,
    chat_history: str,
    doc_tree: object, # TODO: implement PyDantic model for doc tree and add it as an argument in enhanced_rag
    model: ModelProvider,
    max_retries: int = 2
) -> dict:
    """
    Use the LLM to decide if RAG is necessary for the query.
    
    Args:
        query: The user's query
        chat_history: Formatted chat history
        model: The LLM model to use
        max_retries: Maximum number of retries
    
    Returns:
        dict: Decision information including whether RAG is needed,
              a reformulated query if necessary, and a direct response
    """
    task_description = """
You are an AI assistant helping decide whether to search a document database to answer a question.

Please analyze this query and determine:
1. If you need to search for information to answer accurately, or if you can answer directly based on general knowledge
2. If search is needed, provide a reformulated search query that would be most effective for semantic search
"""

    # This is for the prompt generation only - actual validation happens with the Pydantic model
    response_schema = {
        "needs_rag": "boolean",
        "reasoning": "string explaining your decision",
        "reformulated_query": "optimized query for search (if needs_rag is true)",
        "response": "direct response to the user (if needs_rag is false)"
    }
    
    context = {
        "Chat History": chat_history,
        "User Query": query,
        "Document Tree": json.dumps(doc_tree)
    }

    # Get response from the model with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            prompt = create_structured_prompt(
                task_description=task_description,
                response_schema=response_schema,
                context=context,
                last_error=last_error
            )
            
            response_text = await model.genText(prompt)
            
            # Parse and validate using Pydantic
            validated_response, error = parse_and_validate_with_model(
                response_text=response_text,
                model_class=RagNecessityResponse
            )
            
            if error:
                last_error = error
                logger.warning(f"Error parsing LLM response (attempt {attempt+1}/{max_retries}): {error}")
                if attempt == max_retries - 1:
                    # Last attempt failed, use fallback
                    logger.error(f"All attempts failed for RAG necessity decision. Error: {error}")
                    return {
                        "needs_rag": True,
                        "reasoning": f"Error parsing LLM response after {max_retries} attempts, defaulting to RAG",
                        "reformulated_query": query,
                        "response": ""
                    }
                continue
            
            # Extra validation for required fields based on needs_rag
            if validated_response.needs_rag and not validated_response.reformulated_query:
                last_error = "Missing 'reformulated_query' field when needs_rag is true"
                logger.warning(f"Error validating response (attempt {attempt+1}/{max_retries}): {last_error}")
                continue
                
            if not validated_response.needs_rag and not validated_response.response:
                last_error = "Missing 'response' field when needs_rag is false"
                logger.warning(f"Error validating response (attempt {attempt+1}/{max_retries}): {last_error}")
                continue
            
            # Convert Pydantic model to dict
            return validated_response.dict()
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Unexpected error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt failed, use fallback
                logger.error(f"All attempts failed for RAG necessity decision. Error: {str(e)}")
                return {
                    "needs_rag": True,
                    "reasoning": f"Error in RAG necessity decision after {max_retries} attempts, defaulting to RAG",
                    "reformulated_query": query,
                    "response": ""
                }

async def generate_response_with_context(
    original_query: str,
    reformulated_query: str,
    context: str,
    chat_history: str,
    model: ModelProvider,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Generate a response using the retrieved context, original query, and chat history.
    
    Args:
        original_query: The user's original query
        reformulated_query: The reformulated query used for search
        context: The retrieved context from the document database
        chat_history: The formatted chat history
        model: The LLM model to use
        max_retries: Maximum number of retries
    
    Returns:
        Dict with response text and context IDs that were used
    """
    task_description = """
You are a helpful AI assistant. Your task is to answer the user's question based on the provided context.
If the answer cannot be found in the provided context, acknowledge that you don't have the specific information 
but try to provide a helpful response based on your general knowledge.

Guidelines:
- Answer the question directly and specifically based on the provided context
- If the context doesn't contain the answer, say so and provide a general response
- Cite specific documents by their ID when referencing information
- Be concise and accurate in your response
- Try to organize information from multiple sources in a coherent way

Only include document IDs that were actually useful in answering the question.
"""

    # This is for the prompt generation only - actual validation happens with the Pydantic model
    response_schema = {
        "response": "your detailed response to the user",
        "context": ["list of document IDs that were useful"]
    }
    
    context_data = {
        "Chat History": chat_history,
        "User Query": original_query,
        "Context from Knowledge Base": context
    }

    # Get response from the model with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            prompt = create_structured_prompt(
                task_description=task_description, 
                response_schema=response_schema,
                context=context_data,
                last_error=last_error
            )
            
            response_text = await model.genText(prompt)
            
            # Parse and validate using Pydantic
            validated_response, error = parse_and_validate_with_model(
                response_text=response_text,
                model_class=RagResponse
            )
            
            if error:
                last_error = error
                logger.warning(f"Error parsing context response (attempt {attempt+1}/{max_retries}): {error}")
                if attempt == max_retries - 1:
                    # Last attempt failed, use fallback
                    logger.error(f"All attempts failed for generating response with context. Error: {error}")
                    return {
                        "response": "I apologize, but I encountered an error processing your request. Please try asking your question differently.",
                        "context": []
                    }
                continue
            
            # Convert Pydantic model to dict
            return validated_response.model_dump()
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Unexpected error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt failed, use fallback
                logger.error(f"All attempts failed for generating response with context. Error: {str(e)}")
                return {
                    "response": "I apologize, but I encountered an error processing your request. Please try asking your question differently.",
                    "context": []
                } 