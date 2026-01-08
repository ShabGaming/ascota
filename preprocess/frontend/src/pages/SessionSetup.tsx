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
  Text,
  Link,
  Badge,
  Switch,
} from '@chakra-ui/react'
import { AddIcon, DeleteIcon, WarningIcon } from '@chakra-ui/icons'
import { createSession, checkContextStatus } from '../api/client'
import { useSessionStore } from '../state/session'

interface SessionSetupProps {
  onComplete: () => void
}

function SessionSetup({ onComplete }: SessionSetupProps) {
  const [contexts, setContexts] = useState<string[]>([])
  const [currentPath, setCurrentPath] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [preprocessedContexts, setPreprocessedContexts] = useState<Set<string>>(new Set())
  const [overwriteExisting, setOverwriteExisting] = useState(false)
  
  const toast = useToast()
  const setSession = useSessionStore(state => state.setSession)
  
  const handleAddContext = async () => {
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
      
      // Check if context has been preprocessed BEFORE adding to list
      let isPreprocessed = false
      try {
        const status = await checkContextStatus(trimmedPath)
        isPreprocessed = status.is_preprocessed
      } catch {
        // Silently fail - context might not exist yet or file might not be readable
        // This is expected for new contexts
      }
      
      // Add context to list
      setContexts([...contexts, trimmedPath])
      setCurrentPath('')
      
      // If preprocessed, show warning immediately
      if (isPreprocessed) {
        setPreprocessedContexts(prev => new Set(prev).add(trimmedPath))
        toast({
          title: 'Context already preprocessed',
          description: 'This directory has been preprocessed previously. Exporting will update the status.',
          status: 'warning',
          duration: 5000,
          isClosable: true,
        })
      }
    }
  }
  
  const handleRemoveContext = (index: number) => {
    const removedPath = contexts[index]
    setContexts(contexts.filter((_, i) => i !== index))
    // Remove from preprocessed set if it was there
    if (removedPath && preprocessedContexts.has(removedPath)) {
      setPreprocessedContexts(prev => {
        const newSet = new Set(prev)
        newSet.delete(removedPath)
        return newSet
      })
    }
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
        overwrite_existing: overwriteExisting,
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
                    {contexts.map((path, index) => {
                      const isPreprocessed = preprocessedContexts.has(path)
                      return (
                        <ListItem
                          key={index}
                          p={2}
                          bg="gray.50"
                          borderRadius="md"
                          display="flex"
                          justifyContent="space-between"
                          alignItems="center"
                        >
                          <HStack spacing={2} flex={1}>
                            <Box fontSize="sm" fontFamily="mono" flex={1}>{path}</Box>
                            {isPreprocessed && (
                              <Badge colorScheme="orange" display="flex" alignItems="center" gap={1}>
                                <WarningIcon />
                                Already preprocessed
                              </Badge>
                            )}
                          </HStack>
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
                      )
                    })}
                  </List>
                )}
              </FormControl>
            </VStack>
          </CardBody>
        </Card>
        
        <Card>
          <CardBody>
            <FormControl display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <FormLabel mb={1}>Overwrite & Delete Existing Context</FormLabel>
                <Text fontSize="sm" color="gray.600">
                  When enabled, deletes masks folder and preprocess.json from each find's .ascota folder
                </Text>
              </Box>
              <Switch
                isChecked={overwriteExisting}
                onChange={(e) => setOverwriteExisting(e.target.checked)}
                colorScheme="red"
                size="lg"
              />
            </FormControl>
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
        
        {/* Footer */}
        <Box mt={8} pt={6} borderTop="1px" borderColor="gray.200">
          <VStack spacing={3}>
            <Text fontSize="sm" color="gray.600" textAlign="center">
              ASCOTA Preprocess Pipeline | APSAP
            </Text>
            <HStack spacing={4} justify="center" flexWrap="wrap">
              <Link
                href="https://github.com/ShabGaming/ascota"
                isExternal
                fontSize="sm"
                color="blue.500"
                _hover={{ color: 'blue.600', textDecoration: 'underline' }}
              >
                Repository
              </Link>
              <Text fontSize="sm" color="gray.400">•</Text>
              <Link
                href="https://github.com/shabGaming/"
                isExternal
                fontSize="sm"
                color="blue.500"
                _hover={{ color: 'blue.600', textDecoration: 'underline' }}
              >
                GitHub
              </Link>
              <Text fontSize="sm" color="gray.400">•</Text>
              <Link
                href="https://shahabai.com"
                isExternal
                fontSize="sm"
                color="blue.500"
                _hover={{ color: 'blue.600', textDecoration: 'underline' }}
              >
                Website
              </Link>
              <Text fontSize="sm" color="gray.400">•</Text>
              <Link
                href="https://www.linkedin.com/in/shahabai/"
                isExternal
                fontSize="sm"
                color="blue.500"
                _hover={{ color: 'blue.600', textDecoration: 'underline' }}
              >
                LinkedIn
              </Link>
            </HStack>
          </VStack>
        </Box>
      </VStack>
    </Container>
  )
}

export default SessionSetup

