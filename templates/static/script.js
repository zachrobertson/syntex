document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const sendButton = document.getElementById('sendButton');
    const uploadForm = document.getElementById('uploadForm');
    const documentsList = document.getElementById('documentsList');
    const documentContent = document.getElementById('documentContent');
    const chatSessionsList = document.getElementById('chatSessionsList');
    const newSessionButton = document.getElementById('newSessionButton');
    const currentSessionName = document.getElementById('currentSessionName');

    let currentSessionId = null;
    let lastServerStartTime = null;

    // Check server status and reset state if needed
    async function checkServerStatus() {
        try {
            const response = await fetch('/server-status');
            if (!response.ok) {
                throw new Error('Failed to get server status');
            }
            const data = await response.json();
            
            // If this is the first time checking or server has restarted
            if (lastServerStartTime === null || lastServerStartTime !== data.start_time) {
                lastServerStartTime = data.start_time;
                
                // Reset client state
                currentSessionId = null;
                currentSessionName.textContent = '';
                chatMessages.innerHTML = '';
                documentContent.innerHTML = '';
                documentContent.style.display = 'none';
                
                // Reset input container to initial state
                const inputContainer = document.querySelector('.input-container');
                inputContainer.classList.add('initial-state');
                
                // Clear chat sessions list
                chatSessionsList.innerHTML = '';
                
                // If server is in clean mode, show a message
                if (data.clean_mode) {
                    const message = document.createElement('div');
                    message.className = 'server-message';
                    message.textContent = 'Server has been restarted in clean mode. All previous sessions have been removed.';
                    chatMessages.appendChild(message);
                }
            }
            
            // Load current state
            await loadChatSessions();
            await updateDocumentsList();
        } catch (error) {
            console.error('Error checking server status:', error);
        }
    }

    // Load chat sessions
    async function loadChatSessions() {
        try {
            const response = await fetch('/chat-sessions');
            if (!response.ok) {
                throw new Error('Failed to load chat sessions');
            }
            const data = await response.json();
            
            // Clear existing sessions
            chatSessionsList.innerHTML = '';
            
            // Add each session to the list
            data.sessions.forEach(session => {
                const li = document.createElement('li');
                li.dataset.sessionId = session.id;
                
                const nameSpan = document.createElement('span');
                nameSpan.className = 'session-name';
                nameSpan.textContent = session.name;
                
                const dateSpan = document.createElement('span');
                dateSpan.className = 'session-date';
                dateSpan.textContent = new Date(session.updated_at * 1000).toLocaleDateString();
                
                li.appendChild(nameSpan);
                li.appendChild(dateSpan);
                
                li.addEventListener('click', () => {
                    switchToSession(session.id, session.name);
                });
                
                chatSessionsList.appendChild(li);
            });
        } catch (error) {
            console.error('Error loading chat sessions:', error);
        }
    }

    // Create new chat session
    async function createNewSession() {
        const sessionName = prompt('Enter a name for the new chat session:');
        if (!sessionName) return;

        try {
            const response = await fetch('/chat-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: sessionName
                })
            });

            if (!response.ok) {
                throw new Error('Failed to create new session');
            }

            const data = await response.json();
            await loadChatSessions();
            switchToSession(data.id, data.name);
        } catch (error) {
            console.error('Error creating new session:', error);
            alert('Failed to create new chat session');
        }
    }

    // Switch to a different chat session
    async function switchToSession(sessionId, sessionName) {
        currentSessionId = sessionId;
        currentSessionName.textContent = sessionName;
        
        // Update active session in the list
        document.querySelectorAll('#chatSessionsList li').forEach(li => {
            li.classList.toggle('active', li.dataset.sessionId === sessionId.toString());
        });

        // Remove initial state and show header
        document.querySelector('.input-container').classList.remove('initial-state');
        document.querySelector('.chat-header').classList.add('active');

        // Load messages for this session
        try {
            const response = await fetch(`/chat-session/${sessionId}`);
            if (!response.ok) {
                throw new Error('Failed to load session messages');
            }
            const data = await response.json();
            
            // Clear existing messages
            chatMessages.innerHTML = '';
            
            // Add each message to the chat
            data.messages.forEach(msg => {
                addMessage(
                    msg.content,
                    msg.role === 'user',
                    msg.context_ids
                );
            });
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } catch (error) {
            console.error('Error loading session messages:', error);
        }
    }

    // Load chat history from server
    async function loadChatHistory() {
        try {
            const response = await fetch('/chat-history');
            if (!response.ok) {
                throw new Error('Failed to load chat history');
            }
            const data = await response.json();
            
            // Clear existing messages
            chatMessages.innerHTML = '';
            
            // Add each message to the chat
            data.messages.forEach(msg => {
                addMessage(
                    msg.content,
                    msg.role === 'user',
                    msg.context_ids
                );
            });
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    });

    // Handle sending messages
    async function sendMessage() {
        if (!currentSessionId) {
            alert('Please select or create a chat session first');
            return;
        }

        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, true);
        messageInput.value = '';
        messageInput.style.height = 'auto';

        try {
            // Send message to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input: message,
                    session_id: currentSessionId
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get response');
            }

            const data = await response.json();
            addMessage(data.response, false, data.context);
            
            // Clear previous document content
            documentContent.innerHTML = '';
            
            // Reload sessions to update timestamps
            await loadChatSessions();
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, there was an error processing your message.', false);
        }
    }

    // Show document content
    async function showDocumentContent(docId) {
        try {
            const response = await fetch(`/get-document/${docId}`);
            if (!response.ok) {
                throw new Error('Failed to fetch document');
            }
            
            // Create document content display
            const data = await response.json();
            documentContent.innerHTML = `
                <div class="document-header">
                    <h3>Document ${docId}</h3>
                    <button class="close-document">×</button>
                </div>
                <div class="document-text">${data.content}</div>
            `;
            
            // Add close button functionality
            const closeButton = documentContent.querySelector('.close-document');
            closeButton.addEventListener('click', () => {
                documentContent.innerHTML = '';
                documentContent.style.display = 'none';
            });
            
            // Make sure the document content is visible
            documentContent.style.display = 'block';
            
            // Scroll to document content
            documentContent.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error('Error fetching document:', error);
            documentContent.innerHTML = '<p>Error loading document content</p>';
            documentContent.style.display = 'block';
        }
    }

    // Handle message input
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);

    // Handle document uploads
    const fileInput = document.getElementById('documentInput');
    fileInput.addEventListener('change', async (e) => {
        const files = fileInput.files;
        if (files.length === 0) return;

        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        try {
            // Check if this is a directory upload by looking at the first file's webkitRelativePath
            const isDirectory = files[0].webkitRelativePath && files[0].webkitRelativePath.includes('/');
            
            const response = await fetch(isDirectory ? '/add-directory' : '/add-document', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Failed to upload ${isDirectory ? 'directory' : 'documents'}`);
            }

            const result = await response.json();
            updateDocumentsList();
            fileInput.value = ''; // Clear the file input
        } catch (error) {
            console.error(`Error uploading ${isDirectory ? 'directory' : 'documents'}:`, error);
            alert(`Failed to upload ${isDirectory ? 'directory' : 'documents'}. Please try again.`);
        }
    });

    // Update documents list with tree structure
    async function updateDocumentsList() {
        try {
            const response = await fetch('/list-documents');
            if (!response.ok) {
                throw new Error('Failed to fetch documents');
            }

            const data = await response.json();
            documentsList.innerHTML = '';

            function createTreeItem(name, item, isFile = false) {
                const li = document.createElement('li');
                li.className = `document-tree-item ${isFile ? 'file' : 'folder'}`;
                
                if (!isFile) {
                    const toggle = document.createElement('span');
                    toggle.className = 'toggle';
                    li.appendChild(toggle);
                }
                
                const nameSpan = document.createElement('span');
                nameSpan.className = 'name';
                nameSpan.textContent = name;
                li.appendChild(nameSpan);
                
                if (isFile) {
                    li.addEventListener('click', () => {
                        showDocumentContent(item.id);
                    });
                } else {
                    const subList = document.createElement('ul');
                    subList.className = 'document-tree';
                    
                    Object.entries(item).forEach(([key, value]) => {
                        if (key === 'is_file') return;
                        const subItem = createTreeItem(
                            key,
                            value,
                            value.is_file
                        );
                        subList.appendChild(subItem);
                    });
                    
                    li.appendChild(subList);
                    
                    // Toggle folder expansion
                    const toggle = li.querySelector('.toggle');
                    toggle.addEventListener('click', (e) => {
                        e.stopPropagation();
                        li.classList.toggle('expanded');
                    });
                }
                
                return li;
            }

            Object.entries(data.documents).forEach(([name, item]) => {
                const treeItem = createTreeItem(
                    name,
                    item,
                    item.is_file
                );
                documentsList.appendChild(treeItem);
            });
        } catch (error) {
            console.error('Error fetching documents:', error);
        }
    }

    // Initial documents list update
    updateDocumentsList();

    // Load chat history when page loads
    loadChatHistory();

    // Event listeners
    newSessionButton.addEventListener('click', createNewSession);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    sendButton.addEventListener('click', sendMessage);

    // Initial load
    checkServerStatus();

    // Check server status periodically (every 5 seconds)
    setInterval(checkServerStatus, 5000);
});

// Add message to chat
function addMessage(text, isUser, contextIds = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Use marked to parse markdown for the main text
    const formattedText = marked.parse(text);
    contentDiv.innerHTML = formattedText;
    
    // Add context links if this is a bot message and there are context IDs
    if (!isUser && contextIds && contextIds.length > 0) {
        const contextLinks = document.createElement('div');
        contextLinks.className = 'context-links';
        contextLinks.innerHTML = '<span class="context-label">References:</span>';
        
        contextIds.forEach(id => {
            const link = document.createElement('a');
            link.href = '#';
            link.className = 'context-link';
            link.textContent = `Document ${id}`;
            link.dataset.docId = id;
            link.addEventListener('click', (e) => {
                e.preventDefault();
                showDocumentContent(id);
            });
            contextLinks.appendChild(link);
        });
        
        contentDiv.appendChild(contextLinks);
    }
    
    // Add syntax highlighting to code blocks
    contentDiv.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightBlock(block);
    });
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
} 