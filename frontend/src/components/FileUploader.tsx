import { useState, useRef, ChangeEvent, JSX } from 'react';
import { 
  Button,
  Typography,
  Paper,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  CircularProgress
} from '@mui/material';
import { Folder, FileType, CheckCircle } from 'lucide-react';

interface FileItem {
  name: string;
  file: File;
}

interface DirectoryStructure {
  path: string;
  files: FileItem[];
}

export default function DirectoryUpload(): JSX.Element {
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadComplete, setUploadComplete] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = (): void => {
    if (inputRef.current) {
      inputRef.current.click();
    }
  };

  const handleDirectoryUpload = (e: ChangeEvent<HTMLInputElement>): void => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;
    
    setIsUploading(true);
    
    // Convert FileList to array for easier manipulation
    const filesArray = Array.from(fileList);
    
    // In a real application, you might want to upload these files to a server
    // For this example, we'll just display the file structure
    setTimeout(() => {
      setFiles(filesArray);
      setIsUploading(false);
      setUploadComplete(true);
    }, 1500);
  };

  const getFileIcon = (file: File): JSX.Element => {
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    
    if (['jpg', 'jpeg', 'png', 'gif', 'svg'].includes(extension)) {
      return <FileType size={16} color="#2196f3" />;
    } else if (['pdf', 'doc', 'docx', 'txt'].includes(extension)) {
      return <FileType size={16} color="#f44336" />;
    } else if (['js', 'jsx', 'ts', 'tsx', 'html', 'css'].includes(extension)) {
      return <FileType size={16} color="#ffc107" />;
    }
    
    return <FileType size={16} color="#757575" />;
  };

  const getDirectoryStructure = (): DirectoryStructure[] => {
    // Group files by directory path
    const directoryMap: Record<string, FileItem[]> = {};
    
    files.forEach(file => {
      // Get the relative path excluding the file name
      const relativePath = file.webkitRelativePath || (file as any).webkitRelativePath || '';
      
      const pathParts = relativePath.split('/');
      const fileName = pathParts.pop() || file.name;
      const directory = pathParts.join('/');
      
      if (!directoryMap[directory]) {
        directoryMap[directory] = [];
      }
      
      directoryMap[directory].push({
        name: fileName,
        file: file
      });
    });
    
    // Convert the map to an array of directory objects
    const result = Object.keys(directoryMap).map(dir => ({
      path: dir,
      files: directoryMap[dir]
    }));

    return result;
  };

  return (
    <Paper elevation={3} sx={{ maxWidth: 500, mx: 'auto', p: 3 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Directory Upload
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
      
      {files.length > 0 && (
        <Box mt={2}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            Directory Structure
          </Typography>
          <Paper variant="outlined" sx={{ maxHeight: 320, overflow: 'auto', bgcolor: 'transparent' }}>
            {getDirectoryStructure().map((dir, dirIndex) => (
              <Box key={dirIndex} sx={{ mb: 2, px: 2, pt: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', fontWeight: 500 }}>
                  <Folder size={16} color="#f57c00" style={{ marginRight: 8 }} />
                  <Typography variant="body2" fontWeight="medium" color="white">
                    {dir.path || 'Root'}
                  </Typography>
                </Box>
                <List dense disablePadding sx={{ pl: 3 }}>
                  {dir.files.map((file, fileIndex) => {
                    // Debug to check if file names are available
                    const displayName = file.name || file.file.name || 'Unknown file';
                    
                    return (
                      <ListItem key={fileIndex} disablePadding sx={{ py: 0.25 }}>
                        <ListItemIcon sx={{ minWidth: 24 }}>
                          {getFileIcon(file.file)}
                        </ListItemIcon>
                        <ListItemText 
                          primary={
                            <Typography variant="body2" color="white" fontSize="0.875rem">
                              {displayName}
                            </Typography>
                          }
                        />
                      </ListItem>
                    );
                  })}
                </List>
              </Box>
            ))}
          </Paper>
        </Box>
      )}
    </Paper>
  );
}