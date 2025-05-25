import { useState, useEffect, useRef, ChangeEvent, JSX } from 'react';
import { Folder, CheckCircle, FileType, ChevronDown, ChevronRight, Pencil, Check } from 'lucide-react';
import { Box, List, ListItem, ListItemButton, ListItemText, ListItemIcon, Button, Typography, Paper, Alert, CircularProgress, TextField, IconButton } from '@mui/material';
import { ChatSession, Document } from '../types';
import { chatApi, documentApi } from '../services/api';

interface DirectoryStructure {
  path: string;
  files: Document[];
}

interface SidebarProps {
  currentSession: ChatSession | null;
  onSessionSelect: (session: ChatSession) => void;
  onFileSelect?: (documentId: string) => void;
}

const Sidebar = ({ currentSession, onSessionSelect, onFileSelect }: SidebarProps) => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadComplete, setUploadComplete] = useState<boolean>(false);
  const [collapsedFolders, setCollapsedFolders] = useState<Record<string, boolean>>({});
  const [editingSessionId, setEditingSessionId] = useState<number | null>(null);
  const [editingName, setEditingName] = useState<string>('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const loadData = async () => {
      const [sessionsData, documentsData] = await Promise.all([
        chatApi.listSessions(),
        documentApi.list()
      ]);
      setSessions(sessionsData || []);
      setDocuments(documentsData || []);
    };
    loadData();
  }, [currentSession]);

  const handleNewSession = async () => {
    const newSession = await chatApi.createSession(`Chat ${sessions.length + 1}`);
    setSessions([...sessions, newSession]);
    onSessionSelect(newSession);
  };

  const handleUploadClick = (): void => {
    if (inputRef.current) {
      inputRef.current.click();
    }
  };

  const handleDirectoryUpload = async (e: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;
    
    setIsUploading(true);
    setUploadComplete(false);
    
    try {
      // Convert FileList to array for API upload
      const filesArray = Array.from(fileList);
      
      // Upload the files to the backend using documentApi
      await documentApi.uploadDirectory(filesArray);
      
      // Refresh the document list after upload
      const documentsData = await documentApi.list();
      setDocuments(documentsData || []);
      
      setUploadComplete(true);
    } catch (error) {
      console.error("Error uploading files:", error);
      // You might want to add error state handling here
    } finally {
      setIsUploading(false);
    }
  };

  const getDirectoryStructure = (): DirectoryStructure[] => {
    // Group files by directory path
    const directoryMap: Record<string, Document[]> = {};
    
    documents.forEach(document => {
      // Get the relative path excluding the file name
      const directory = document.path || '';  
      if (!directoryMap[directory]) {
        directoryMap[directory] = [];
      }
      
      directoryMap[directory].push(document);
    });
    
    // Convert the map to an array of directory objects
    const result = Object.keys(directoryMap).map(dir => ({
      path: dir,
      files: directoryMap[dir]
    }));

    return result;
  };

  const getFileIcon = (name: string): JSX.Element => {
    const extension = name.split('.').pop()?.toLowerCase() || '';
    
    if (['jpg', 'jpeg', 'png', 'gif', 'svg'].includes(extension)) {
      return <FileType size={16} color="#2196f3" />;
    } else if (['pdf', 'doc', 'docx', 'txt'].includes(extension)) {
      return <FileType size={16} color="#f44336" />;
    } else if (['js', 'jsx', 'ts', 'tsx', 'html', 'css'].includes(extension)) {
      return <FileType size={16} color="#ffc107" />;
    }
    
    return <FileType size={16} color="#757575" />;
  };

  const handleFileClick = (document: Document) => {
    if (onFileSelect) {
      onFileSelect(document.id);
    }
  };

  const toggleFolder = (path: string) => {
    setCollapsedFolders(prev => ({
      ...prev,
      [path]: !prev[path]
    }));
  };

  const handleEditClick = (session: ChatSession) => {
    setEditingSessionId(session.id);
    setEditingName(session.name);
    // Focus the input after it's rendered
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
        inputRef.current.select();
      }
    }, 0);
  };

  const handleEditSubmit = async () => {
    if (editingSessionId && editingName.trim()) {
      try {
        const updatedSession = await chatApi.renameSession(editingSessionId, editingName.trim());
        setSessions(sessions.map(s => s.id === updatedSession.id ? updatedSession : s));
        if (currentSession?.id === updatedSession.id) {
          onSessionSelect(updatedSession);
        }
        setEditingSessionId(null);
      } catch (error) {
        console.error('Error renaming session:', error);
      }
    }
  };

  const handleEditCancel = () => {
    setEditingSessionId(null);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleEditSubmit();
    } else if (e.key === 'Escape') {
      handleEditCancel();
    }
  };

  return (
    <Box sx={{ width: 300, height: '100%', borderRight: '1px solid #333' }}>
      <Paper elevation={0} sx={{ p: 2, height: '100%' }}>
        {/*Chat Session Manager*/}
        <Typography variant="h6" gutterBottom>
          Chat Sessions
        </Typography>
        <Button
          variant="contained"
          fullWidth
          onClick={handleNewSession}
          sx={{ mb: 2 }}
        >
          New Chat
        </Button>
        <List>
          {sessions.map((session) => (
            <ListItem 
              key={session.id} 
              disablePadding
              secondaryAction={
                editingSessionId === session.id ? (
                  <IconButton 
                    edge="end" 
                    onClick={handleEditSubmit}
                    size="small"
                  >
                    <Check size={16} />
                  </IconButton>
                ) : (
                  <IconButton 
                    edge="end" 
                    onClick={() => handleEditClick(session)}
                    size="small"
                  >
                    <Pencil size={16} />
                  </IconButton>
                )
              }
            >
              <ListItemButton
                selected={currentSession?.id === session.id}
                onClick={() => {
                  if (editingSessionId !== session.id) {
                    onSessionSelect(session);
                  }
                }}
                sx={{ pr: 6 }} // Make room for the icon button
              >
                {editingSessionId === session.id ? (
                  <TextField
                    inputRef={inputRef}
                    value={editingName}
                    onChange={(e) => setEditingName(e.target.value)}
                    onKeyDown={handleEditKeyDown}
                    onBlur={handleEditCancel}
                    size="small"
                    fullWidth
                    autoFocus
                    sx={{
                      '& .MuiInputBase-root': {
                        color: 'inherit',
                        '&:before, &:after': {
                          display: 'none'
                        }
                      }
                    }}
                  />
                ) : (
                  <ListItemText primary={session.name} />
                )}
              </ListItemButton>
            </ListItem>
          ))}
        </List>

        {/* Document Upload Section */}
        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          Documents
        </Typography>
        <input
          type="file"
          ref={inputRef}
          onChange={handleDirectoryUpload}
          style={{ display: 'none' }}
          // @ts-ignore - TypeScript doesn't recognize these directory attributes natively
          webkitdirectory=""
          directory=""
          mozdirectory=""
          multiple
        />
        
        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={handleUploadClick}
          disabled={isUploading}
          startIcon={isUploading ? <CircularProgress size={20} color="inherit" /> : <Folder size={20} />}
          sx={{ mb: 2 }}
        >
          {isUploading ? 'Uploading...' : 'Choose Directory'}
        </Button>
        
        {uploadComplete && (
          <Alert 
            icon={<CheckCircle fontSize="inherit" />} 
            severity="success" 
            sx={{ mb: 2 }}
          >
            Directory uploaded successfully!
          </Alert>
        )}
        {documents.length > 0 && (
          <Box mt={2}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Directory Structure
            </Typography>
            <Paper 
              variant="outlined" 
              sx={{ 
                maxHeight: 320, 
                overflow: 'auto', 
                bgcolor: 'transparent', 
                width: "100%"
              }}
            >
              {getDirectoryStructure().map((dir, dirIndex) => {
                const isCollapsed = collapsedFolders[dir.path] ?? false;
                return (
                  <Box key={dirIndex} sx={{ mb: 2, px: 2, pt: 1 }}>
                    <Box 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        fontWeight: 500,
                        overflow: 'hidden',
                        cursor: 'pointer',
                        '&:hover': {
                          opacity: 0.8
                        }
                      }}
                      onClick={() => toggleFolder(dir.path)}
                    >
                      <Box sx={{ 
                        minWidth: 32, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center' 
                      }}>
                        {isCollapsed ? <ChevronRight size={16} color="#f57c00" /> : <ChevronDown size={16} color="#f57c00" />}
                      </Box>
                      <Box sx={{ 
                        minWidth: 32, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center' 
                      }}>
                        <Folder size={16} color="#f57c00" />
                      </Box>
                      <Typography 
                        variant="body2" 
                        fontWeight="medium" 
                        color="white"
                        sx={{
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {dir.path || "Root"}
                      </Typography>
                    </Box>
                    {!isCollapsed && (
                      <List dense disablePadding sx={{ pl: 3 }}>
                        {dir.files.map((file, fileIndex) => {                      
                          return (
                            <ListItemButton
                              key={fileIndex}
                              dense
                              sx={{ py: 0.25 }}
                              onClick={() => handleFileClick(file)}
                            >
                              <ListItemIcon sx={{ minWidth: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                {getFileIcon(file.filename)}
                              </ListItemIcon>
                              <ListItemText 
                                primary={
                                  <Typography 
                                    variant="body2" 
                                    color="white" 
                                    fontSize="0.875rem"
                                    sx={{
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap'
                                    }}
                                  >
                                    {file.filename}
                                  </Typography>
                                }
                              />
                            </ListItemButton>
                          );
                        })}
                      </List>
                    )}
                  </Box>
                );
              })}
            </Paper>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default Sidebar; 