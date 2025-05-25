import { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { Box, Typography, TextField, Button, Paper } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import { ChatSession, ChatMessage } from '../types';
import { chatApi } from '../services/api';
import DocumentWindow from './DocumentWindow';

interface ChatContainerProps {
  currentSession: ChatSession | null;
  onSessionSelect?: (session: ChatSession) => void;
}

export interface ChatContainerHandle {
  handleFileSelect: (documentId: string) => void;
}

const ChatContainer = forwardRef<ChatContainerHandle, ChatContainerProps>(({ currentSession, onSessionSelect }, ref) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);

  // Expose the handleFileSelect method via ref
  useImperativeHandle(ref, () => ({
    handleFileSelect: (documentId: string) => {
      setSelectedDocumentId(documentId);
    }
  }));

  useEffect(() => {
    if (currentSession) {
      const loadMessages = async () => {
        const history = await chatApi.getHistory(currentSession.id);
        setMessages(history.messages);
      };
      loadMessages();
    } else {
      setMessages([]);
    }
  }, [currentSession]);

  const handleSend = async () => {
    if (!input.trim()) return;

    let sessionToUse = currentSession;

    // If no session is selected, create a new one
    if (!sessionToUse) {
      try {
        const sessions = await chatApi.listSessions();
        const lastSessionId = sessions.length > 0 ? Math.max(...sessions.map((session) => session.id)): 0;
        const newSession = await chatApi.createSession(`Chat ${lastSessionId + 1}`);
        if (onSessionSelect) {
          onSessionSelect(newSession);
        }
        sessionToUse = newSession;
      } catch (error) {
        console.error('Error creating new session:', error);
        return;
      }
    }

    const userMessage: ChatMessage = {
      id: Date.now(),
      session_id: sessionToUse.id,
      content: input,
      role: 'user',
      created_at: new Date().toISOString(),
    };

    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');

    try {
      const response = await chatApi.sendMessage(sessionToUse.id, input);
      setMessages(prevMessages => [...prevMessages, response]);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const handleContextButtonClick = (contextId: string) => {
    if (contextId) {
      setSelectedDocumentId(contextId);
    }
  };

  // Function to close the document window
  const handleCloseDocument = () => {
    setSelectedDocumentId(null);
  };

  const renderMessages = () => {
    if (!currentSession) {
      return (
        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="h6" color="text.secondary">
            Type a message to start a new chat
          </Typography>
        </Box>
      );
    }

    return messages && messages.length > 0 ? (
      messages.map((message) => (
        <Paper
          key={message.id}
          elevation={0}
          sx={{
            p: 2,
            mb: 2,
            backgroundColor: message.role === 'user' ? 'primary.main' : 'background.paper',
            color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
            maxWidth: '80%',
            ml: message.role === 'user' ? 'auto' : 0,
            position: 'relative',
          }}
        >
          <ReactMarkdown>{message.content}</ReactMarkdown>
          
          {message.role === 'assistant' && message.context_ids && message.context_ids.length > 0 && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
              {message.context_ids.map((docRef, index) => (
                <Button
                  key={index}
                  size="small"
                  variant="outlined"
                  onClick={() => handleContextButtonClick(docRef.id)}
                  sx={{
                    fontSize: '0.75rem',
                    textTransform: 'none'
                  }}
                  title={docRef.path ? `${docRef.path}/${docRef.filename}` : docRef.filename}
                >
                  {docRef.path ? `${docRef.path}/${docRef.filename}` : docRef.filename}
                </Button>
              ))}
            </Box>
          )}
        </Paper>
      ))
    ) : (
      <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
        No messages yet. Start the conversation!
      </Typography>
    );
  };

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%' }}>      
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {renderMessages()}
      </Box>
      {/* Document Window */}
      {selectedDocumentId && (
        <DocumentWindow 
          documentId={selectedDocumentId} 
          onClose={handleCloseDocument} 
        />
      )}
      {/*User Input Container*/}
      <Box sx={{ p: 2, borderTop: '1px solid #333' }}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDownCapture={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={currentSession ? "Type your message..." : "Type a message to start a new chat..."}
          />
          <Button
            variant="contained"
            onClick={handleSend}
          >
            Send
          </Button>
        </Box>
      </Box>
    </Box>
  );
});

export default ChatContainer; 