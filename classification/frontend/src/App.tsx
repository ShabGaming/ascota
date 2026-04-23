import { useState } from 'react'
import { Box } from '@chakra-ui/react'
import SessionSetup from './pages/SessionSetup'
import TypeClassification from './pages/TypeClassification'
import PotteryClassification from './pages/PotteryClassification'
import DecorationClassification from './pages/DecorationClassification'
import ColorClassification from './pages/ColorClassification'
import TextureClassification from './pages/TextureClassification'
import { useSessionStore } from './state/session'

function App() {
  const [stage, setStage] = useState<'setup' | 'classification'>('setup')
  const sessionId = useSessionStore((s) => s.sessionId)
  const classificationType = useSessionStore((s) => s.classificationType)

  const handleSessionCreated = () => setStage('classification')
  const handleReset = () => {
    useSessionStore.getState().reset()
    setStage('setup')
  }

  return (
    <Box minH="100vh" bg="gray.50">
      {stage === 'setup' && <SessionSetup onComplete={handleSessionCreated} />}
      {stage === 'classification' && sessionId && classificationType === 'pottery' && (
        <PotteryClassification sessionId={sessionId} onReset={handleReset} />
      )}
      {stage === 'classification' && sessionId && classificationType === 'type' && (
        <TypeClassification sessionId={sessionId} onReset={handleReset} />
      )}
      {stage === 'classification' && sessionId && classificationType === 'decoration' && (
        <DecorationClassification sessionId={sessionId} onReset={handleReset} />
      )}
      {stage === 'classification' && sessionId && classificationType === 'color' && (
        <ColorClassification sessionId={sessionId} onReset={handleReset} />
      )}
      {stage === 'classification' && sessionId && classificationType === 'texture' && (
        <TextureClassification sessionId={sessionId} onReset={handleReset} />
      )}
    </Box>
  )
}

export default App
