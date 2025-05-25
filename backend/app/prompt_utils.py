import json
from typing import Dict, Any, Optional, List, Union, Tuple, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

def create_structured_prompt(
    task_description: str,
    response_schema: Dict[str, Any],
    context: Dict[str, Any],
    last_error: Optional[str] = None
) -> str:
    """
    Create a consistent structured prompt for LLM requests.
    
    Args:
        task_description: Description of the task for the LLM
        response_schema: JSON schema describing the expected response format
        context: Dictionary containing context elements to include in the prompt
        last_error: Optional error message from previous generation attempt
    
    Returns:
        str: Formatted prompt string
    """
    # Format the context elements
    context_str = ""
    for key, value in context.items():
        if value:
            context_str += f"\n{key}:\n{value}\n"
    
    # Format the response schema
    schema_str = json.dumps(response_schema, indent=2)
    
    # Add error context if provided
    error_context = f"\nPrevious error: {last_error}\nPlease fix the error in your response." if last_error else ""
    
    # Create the prompt
    prompt = f"""
{task_description}
{context_str}
{error_context}

Respond with a JSON formatted object with the following schema:
{schema_str}

Do not add any markdown quotes around the JSON object and be careful to escape any quotes or brackets correctly so that the string can be parsed in python using json.loads().
"""
    
    return prompt

def parse_and_validate_with_model(
    response_text: str,
    model_class: Type[T]
) -> Tuple[Optional[T], Optional[str]]:
    """
    Parse JSON from response text and validate using a Pydantic model.
    
    Args:
        response_text: Raw text response from the model
        model_class: Pydantic model class to use for validation
        
    Returns:
        Tuple of (validated_model, error_message)
    """
    try:
        # Try to parse the JSON
        response_json = json.loads(response_text)
        
        # Validate with Pydantic
        validated_model = model_class(**response_json)
        return validated_model, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except ValidationError as e:
        return None, f"Validation error: {str(e)}" 