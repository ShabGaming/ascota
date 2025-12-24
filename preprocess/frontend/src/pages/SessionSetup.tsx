import { useState } from 'react'
import {
  Box,
  Button,
  Container,
  FormControl,
  FormLabel,
  Heading,
  Input,
  VStack,
  HStack,
  IconButton,
  List,
  ListItem,
  useToast,
  Card,
  CardBody,
} from '@chakra-ui/react'
import { AddIcon, DeleteIcon } from '@chakra-ui/icons'
import { createSession } from '../api/client'
import { useSessionStore } from '../state/session'

interface SessionSetupProps {
  onComplete: () => void
}

function SessionSetup({ onComplete }: SessionSetupProps) {
  const [contexts, setContexts] = useState<string[]>([])
  const [currentPath, setCurrentPath] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  
  const toast = useToast()
  const setSession = useSessionStore(state => state.setSession)
  
  const handleAddContext = () => {
    const trimmedPath = currentPath.trim()
    if (trimmedPath) {
      const normalizedPath = trimmedPath.replace(/\\/g, '/').toLowerCase()
      const isDuplicate = contexts.some(ctx => 
        ctx.replace(/\\/g, '/').toLowerCase() === normalizedPath
      )
      
      if (isDuplicate) {
        toast({
          title: 'Duplicate context',
          description: 'This directory has already been added',
          status: 'warning',
          duration: 3000,
        })
        return
      }
      
      setContexts([...contexts, trimmedPath])
      setCurrentPath('')
    }
  }
  
  const handleRemoveContext = (index: number) => {
    setContexts(contexts.filter((_, i) => i !== index))
  }
  
  const handleStart = async () => {
    if (contexts.length === 0) {
      toast({
        title: 'No contexts provided',
        description: 'Please add at least one context path',
        status: 'warning',
        duration: 3000,
      })
      return
    }
    
    setIsLoading(true)
    
    try {
      const response = await createSession({
        contexts,
      })
      
      setSession(response.session_id)
      
      toast({
        title: 'Session created',
        description: 'Starting preprocessing...',
        status: 'success',
        duration: 2000,
      })
      
      onComplete()
    } catch (error) {
      toast({
        title: 'Failed to create session',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      })
      setIsLoading(false)
    }
  }
  
  return (
    <Container maxW="container.lg" py={10}>
      <VStack spacing={8} align="stretch">
        <Box textAlign="center">
          <Heading size="xl" mb={2}>Preprocess Pipeline</Heading>
        </Box>
        
        <Card>
          <CardBody>
            <VStack spacing={6} align="stretch">
              <FormControl>
                <FormLabel>Context Directories</FormLabel>
                <HStack>
                  <Input
                    placeholder="D:\ararat\data\files\N\38\478020\4419550\1"
                    value={currentPath}
                    onChange={(e) => setCurrentPath(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddContext()}
                  />
                  <IconButton
                    aria-label="Add context"
                    icon={<AddIcon />}
                    onClick={handleAddContext}
                    colorScheme="brand"
                  />
                </HStack>
                
                {contexts.length > 0 && (
                  <List spacing={2} mt={4}>
                    {contexts.map((path, index) => (
                      <ListItem
                        key={index}
                        p={2}
                        bg="gray.50"
                        borderRadius="md"
                        display="flex"
                        justifyContent="space-between"
                        alignItems="center"
                      >
                        <Box fontSize="sm" fontFamily="mono" flex={1}>{path}</Box>
                        <IconButton
                          aria-label="Remove context"
                          icon={<DeleteIcon />}
                          size="sm"
                          variant="ghost"
                          colorScheme="red"
                          onClick={() => handleRemoveContext(index)}
                          ml={2}
                        />
                      </ListItem>
                    ))}
                  </List>
                )}
              </FormControl>
            </VStack>
          </CardBody>
        </Card>
        
        <Button
          colorScheme="brand"
          size="lg"
          onClick={handleStart}
          isLoading={isLoading}
          loadingText="Creating session..."
        >
          Start Preprocessing
        </Button>
      </VStack>
    </Container>
  )
}

export default SessionSetup

