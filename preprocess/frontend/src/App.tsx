import { useState, useEffect } from 'react'
import { Box, Container } from '@chakra-ui/react'
import SessionSetup from './pages/SessionSetup'
import Stage1Cards from './pages/Stage1Cards'
import Stage2Mask from './pages/Stage2Mask'
import Stage3Scale from './pages/Stage3Scale'
import { useSessionStore } from './state/session'

function App() {
  const [stage, setStage] = useState<'setup' | 1 | 2 | 3>('setup')
  const sessionId = useSessionStore(state => state.sessionId)
  const currentStage = useSessionStore(state => state.currentStage)
  const setCurrentStage = useSessionStore(state => state.setCurrentStage)
  
  // Sync stage with store
  useEffect(() => {
    if (sessionId && stage === 'setup') {
      setStage(currentStage)
    }
  }, [sessionId, currentStage])
  
  const handleSessionCreated = () => {
    setStage(1)
    setCurrentStage(1)
  }
  
  const handleStage1Complete = () => {
    setStage(2)
    setCurrentStage(2)
  }
  
  const handleStage2Complete = () => {
    setStage(3)
    setCurrentStage(3)
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
      
      {stage === 1 && sessionId && (
        <Stage1Cards
          sessionId={sessionId}
          onComplete={handleStage1Complete}
          onReset={handleReset}
        />
      )}
      
      {stage === 2 && sessionId && (
        <Stage2Mask
          sessionId={sessionId}
          onComplete={handleStage2Complete}
          onReset={handleReset}
        />
      )}
      
      {stage === 3 && sessionId && (
        <Stage3Scale
          sessionId={sessionId}
          onComplete={handleReset}
          onReset={handleReset}
        />
      )}
    </Box>
  )
}

export default App

