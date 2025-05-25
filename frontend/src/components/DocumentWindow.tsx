import { useState, useEffect } from 'react';
import { Box, Paper, Typography, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { Document } from '../types';
import { documentApi } from '../services/api';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface DocumentWindowProps {
  documentId: string | null;
  onClose: () => void;
}

const DocumentWindow = ({ documentId, onClose }: DocumentWindowProps) => {
  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Function to determine language based on file extension
  const getLanguage = (filename: string): string => {
    const extension = filename.split('.').pop()?.toLowerCase() || '';
    
    const extensionMap: {[key: string]: string} = {
      'py': 'python',
      'js': 'javascript',
      'ts': 'typescript',
      'tsx': 'tsx',
      'jsx': 'jsx',
      'html': 'html',
      'css': 'css',
      'json': 'json',
      'md': 'markdown',
      'sql': 'sql',
      'sh': 'bash',
      'yml': 'yaml',
      'yaml': 'yaml',
      'java': 'java',
      'c': 'c',
      'cpp': 'cpp',
      'cs': 'csharp',
      'go': 'go',
      'rb': 'ruby',
      'php': 'php',
      'rs': 'rust',
      'swift': 'swift',
      'kt': 'kotlin'
    };
    
    return extensionMap[extension] || 'text';
  };

  useEffect(() => {
    if (!documentId) {
      setDocument(null);
      return;
    }

    const fetchDocument = async () => {
      setLoading(true);
      setError(null);
      try {
        const doc = await documentApi.get(documentId);
        setDocument(doc);
      } catch (err) {
        console.error('Error fetching document:', err);
        setError('Failed to load document');
      } finally {
        setLoading(false);
      }
    };

    fetchDocument();
  }, [documentId]);

  if (!documentId) return null;

  return (
    <Box sx={{ position: 'relative', mb: 2 }}>
      <Paper
        elevation={3}
        sx={{
          p: 2,
          maxHeight: '40vh',
          overflowY: 'auto',
          width: '100%',
          borderLeft: '4px solid #90caf9',
          position: 'relative'
        }}
      >
        {loading && (
          <Typography variant="body2" color="text.secondary">
            Loading document...
          </Typography>
        )}

        {error && (
          <Typography variant="body2" color="error">
            {error}
          </Typography>
        )}

        {document && (
          <>
            <Typography 
              variant="subtitle1" 
              fontWeight="bold" 
              sx={{ mb: 1, pr: 4, wordBreak: 'break-word' }}
            >
              {document.filename}
              <Typography 
                component="span" 
                variant="body2" 
                color="text.secondary" 
                sx={{ ml: 1 }}
              >
                {document.path}
              </Typography>
            </Typography>
            
            <Box sx={{ mt: 2 }}>
              <SyntaxHighlighter 
                language={getLanguage(document.filename)}
                style={atomDark}
                customStyle={{
                  borderRadius: '4px',
                  fontSize: '0.875rem',
                  margin: 0,
                  maxWidth: '100%',
                  overflowX: 'hidden'
                }}
                wrapLines={true}
                wrapLongLines={true}
                lineProps={{style: {wordBreak: 'break-all', whiteSpace: 'pre-wrap'}}}
              >
                {document.content}
              </SyntaxHighlighter>
            </Box>
          </>
        )}
      </Paper>
      
      <Box 
        sx={{ 
          position: 'absolute', 
          top: 8, 
          right: 8, 
          zIndex: 1000,
          backgroundColor: 'rgba(255, 255, 255, 0.8)', 
          borderRadius: '50%',
          boxShadow: '0 0 4px rgba(0,0,0,0.2)'
        }}
      >
        <IconButton onClick={onClose} size="small">
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>
    </Box>
  );
};

export default DocumentWindow;
