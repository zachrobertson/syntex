export interface Document {
    id: string;
    path: string;
    filename: string;
    content: string;
    uploaded_at: number;
}

export interface ChatSession {
    id: number;
    name: string;
    created_at: number;
    updated_at: number;
}

export interface DocumentReference {
    id: string;
    filename: string;
    path: string;
}

export interface RawChatMessage {
    id: number;
    session_id: number;
    content: string;
    role: 'user' | 'assistant';
    created_at: string;
    context_ids?: string; // JSON parsable array of context refs
}

export interface ChatMessage {
    id: number;
    session_id: number;
    content: string;
    role: 'user' | 'assistant';
    created_at: string;
    context_ids?: DocumentReference[];
}

// export interface ChatContext {}

export interface ChatResponse {
    response: string;
    context: string[];
}

export interface SearchQuery {
    query: string;
    limit?: number;
}

export interface ApiResponse<T> {
    data: T;
    error?: string;
} 