import { useState, useRef } from 'react'
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import Sidebar from './components/Sidebar'
import ChatContainer from './components/ChatContainer'
import { ChatSession } from './types'

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
})

function App() {
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const chatContainerRef = useRef<any>(null)

  const handleFileSelect = (documentId: string) => {
    if (chatContainerRef.current?.handleFileSelect) {
      chatContainerRef.current.handleFileSelect(documentId)
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh' }}>
        <Sidebar 
          currentSession={currentSession}
          onSessionSelect={setCurrentSession}
          onFileSelect={handleFileSelect}
        />
        <ChatContainer 
          ref={chatContainerRef}
          currentSession={currentSession}
          onSessionSelect={setCurrentSession}
        />
      </Box>
    </ThemeProvider>
  )
}

export default App
