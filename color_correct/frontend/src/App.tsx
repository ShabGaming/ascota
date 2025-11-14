import { useState } from 'react'
import { Box, Container } from '@chakra-ui/react'
import SessionSetup from './pages/SessionSetup'
import ClusterBoard from './pages/ClusterBoard'
import { useSessionStore } from './state/session'

function App() {
  const [stage, setStage] = useState<'setup' | 'clustering' | 'board'>('setup')
  const sessionId = useSessionStore(state => state.sessionId)
  
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

