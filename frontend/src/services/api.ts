import axios from 'axios';
import { Document, ChatSession, SearchQuery, ChatMessage, RawChatMessage } from '../types';

interface ApiResponse<T> {
    data: T;
    error?: string;
    meta?: Record<string, unknown>;
}

const API_BASE_URL = 'http://localhost:8080';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const documentApi = {
    list: async (): Promise<Document[]> => {
        const response = await api.get<ApiResponse<Document[]>>('/list-documents');
        return response.data.data;
    },

    get: async (id: string): Promise<Document> => {
        const response = await api.get<ApiResponse<Document>>(`/get-document/${id}`);
        return response.data.data;
    },

    upload: async (files: File[]): Promise<Document[]> => {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        const response = await api.post<ApiResponse<Document[]>>('/add-document', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data.data;
    },

    uploadDirectory: async (files: File[]): Promise<Document[]> => {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        const response = await api.post<ApiResponse<Document[]>>('/add-directory', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data.data;
    },

    remove: async (id: string): Promise<void> => {
        await api.delete(`/remove-document/${id}`);
    },
};

// Helper function to transform RawChatMessage to ChatMessage
const formatChatMessage = (rawMessage: RawChatMessage): ChatMessage => ({
    ...rawMessage,
    context_ids: rawMessage.context_ids ? JSON.parse(rawMessage.context_ids) : undefined
});

// Helper function to transform arrays of RawChatMessage to ChatMessage
const formatChatMessages = (rawMessages: RawChatMessage[]): ChatMessage[] => {
    return rawMessages.map(formatChatMessage);
};

export const chatApi = {
    listSessions: async (): Promise<ChatSession[]> => {
        const response = await api.get<ApiResponse<ChatSession[]>>('/chat-sessions');
        return response.data.data;
    },

    createSession: async (name: string): Promise<ChatSession> => {
        const response = await api.post<ApiResponse<ChatSession>>('/chat-session', { name });
        return response.data.data;
    },

    getSession: async (id: number): Promise<{ session: ChatSession, messages: ChatMessage[] }> => {
        const response = await api.get<ApiResponse<{ session: ChatSession, messages: RawChatMessage[] }>>(`/chat-session/${id}`);
        return {
            session: response.data.data.session,
            messages: formatChatMessages(response.data.data.messages)
        };
    },

    renameSession: async (id: number, newName: string): Promise<ChatSession> => {
        const response = await api.patch<ApiResponse<ChatSession>>(`/chat-session/${id}/rename`, { new_name: newName });
        return response.data.data;
    },

    sendMessage: async (sessionId: number, message: string): Promise<ChatMessage> => {
        const response = await api.post<ApiResponse<RawChatMessage>>('/chat', {
            input: message,
            session_id: sessionId,
        });
        return formatChatMessage(response.data.data);
    },

    getHistory: async (sessionId: number): Promise<{ messages: ChatMessage[] }> => {
        const response = await api.get<ApiResponse<{ messages: RawChatMessage[] }>>('/chat-history', {
            params: { id: sessionId },
        });
        return { 
            messages: formatChatMessages(response.data.data.messages)
        };
    },
};

export const searchApi = {
    search: async (query: SearchQuery): Promise<any> => {
        const response = await api.post<ApiResponse<any>>('/search', query);
        return response.data.data;
    },
}; 