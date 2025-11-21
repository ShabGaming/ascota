import { useState, useEffect } from 'react'
import { Box, Container } from '@chakra-ui/react'
import SessionSetup from './pages/SessionSetup'
import ClusterBoard from './pages/ClusterBoard'
import { useSessionStore } from './state/session'
import { listSessions, restoreSession } from './api/client'
import { useQuery } from 'react-query'

function App() {
  const [stage, setStage] = useState<'setup' | 'clustering' | 'board'>('setup')
  const sessionId = useSessionStore(state => state.sessionId)
  const setSession = useSessionStore(state => state.setSession)
  
  // Check for active sessions on mount
  useEffect(() => {
    const checkActiveSession = async () => {
      try {
        const sessions = await listSessions()
        // If there's an active session in memory, use it
        if (sessionId) {
          setStage('board')
          return
        }
        // Otherwise, check if there's a most recent persisted session
        if (sessions.persisted_sessions && sessions.persisted_sessions.length > 0) {
          // Optionally auto-restore the most recent session
          // For now, we'll let the user choose from the sessions tab
        }
      } catch (error) {
        console.error('Failed to check for active sessions:', error)
      }
    }
    
    checkActiveSession()
  }, [])
  
  const handleSessionCreated = () => {
    setStage('clustering')
  }
  
  const handleClusteringComplete = () => {
    setStage('board')
  }
  
  const handleReset = () => {
    useSessionStore.getState().reset()
    setStage('setup')
  }
  
  // Update stage when sessionId changes
  useEffect(() => {
    if (sessionId && stage === 'setup') {
      setStage('board')
    }
  }, [sessionId])
  
  return (
    <Box minH="100vh" bg="gray.50">
      {stage === 'setup' && (
        <SessionSetup onComplete={handleSessionCreated} />
      )}
      
      {(stage === 'clustering' || stage === 'board') && sessionId && (
        <ClusterBoard
          sessionId={sessionId}
          onClusteringComplete={handleClusteringComplete}
          onReset={handleReset}
          isLoadingClusters={stage === 'clustering'}
        />
      )}
    </Box>
  )
}

export default App

