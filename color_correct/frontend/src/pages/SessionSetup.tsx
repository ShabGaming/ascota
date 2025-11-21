import { useState, useEffect } from 'react'
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
  Switch,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Text,
  IconButton,
  List,
  ListItem,
  useToast,
  Card,
  CardBody,
  Select,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  Code,
} from '@chakra-ui/react'
import { AddIcon, DeleteIcon } from '@chakra-ui/icons'
import { createSession, listSessions, restoreSession, SessionInfo, deleteSession } from '../api/client'
import { useSessionStore } from '../state/session'
import { useQuery, useMutation, useQueryClient } from 'react-query'

interface SessionSetupProps {
  onComplete: () => void
}

function SessionSetup({ onComplete }: SessionSetupProps) {
  const [contexts, setContexts] = useState<string[]>([])
  const [currentPath, setCurrentPath] = useState('')
  const [overwrite, setOverwrite] = useState(false)
  const [useCustomK, setUseCustomK] = useState(false)
  const [customK, setCustomK] = useState(3)
  const [sensitivity, setSensitivity] = useState(1.0)
  const [previewResolution, setPreviewResolution] = useState(1500)
  const [isLoading, setIsLoading] = useState(false)
  
  const toast = useToast()
  const setSession = useSessionStore(state => state.setSession)
  const queryClient = useQueryClient()
  
  // Fetch sessions list
  const { data: sessionsData, refetch: refetchSessions } = useQuery(
    ['sessions'],
    listSessions,
    {
      refetchOnMount: true,
    }
  )
  
  // Restore session mutation
  const restoreMutation = useMutation(
    (sessionId: string) => restoreSession(sessionId),
    {
      onSuccess: (data) => {
        setSession(data.session_id)
        toast({
          title: 'Session restored',
          description: 'Session has been restored from disk',
          status: 'success',
          duration: 3000,
        })
        onComplete()
      },
      onError: (error) => {
        toast({
          title: 'Failed to restore session',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  // Delete session mutation
  const deleteMutation = useMutation(
    (sessionId: string) => deleteSession(sessionId),
    {
      onSuccess: () => {
        // Always refresh the list, even if session wasn't found (idempotent delete)
        refetchSessions()
        toast({
          title: 'Session deleted',
          status: 'success',
          duration: 2000,
        })
      },
      onError: (error) => {
        // Even on error, try to refresh the list to remove invalid sessions
        refetchSessions()
        toast({
          title: 'Session deleted',
          description: 'Session removed from list (may not have existed)',
          status: 'info',
          duration: 3000,
        })
      },
    }
  )
  
  const handleAddContext = () => {
    if (currentPath.trim()) {
      setContexts([...contexts, currentPath.trim()])
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
        options: {
          image_source: "raw_mode", // Always use RAW mode
          overwrite,
          custom_k: useCustomK ? customK : undefined,
          sensitivity,
          preview_resolution: previewResolution,
        },
      })
      
      setSession(response.session_id)
      
      toast({
        title: 'Session created',
        description: 'Starting clustering...',
        status: 'success',
        duration: 2000,
      })
      
      refetchSessions()
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
  
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString)
      return date.toLocaleString()
    } catch {
      return dateString
    }
  }
  
  return (
    <Container maxW="container.lg" py={10}>
      <VStack spacing={8} align="stretch">
        <Box textAlign="center">
          <Heading size="xl" mb={2}>Color Correction Tool</Heading>
          <Text color="gray.600">
            Batch color correction with intelligent clustering
          </Text>
        </Box>
        
        <Tabs>
          <TabList>
            <Tab>New Session</Tab>
            <Tab>Sessions ({sessionsData?.persisted_sessions?.length || 0})</Tab>
          </TabList>
          
          <TabPanels>
            <TabPanel>
              <VStack spacing={8} align="stretch">
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
                        <Text fontSize="sm" fontFamily="mono">{path}</Text>
                        <IconButton
                          aria-label="Remove context"
                          icon={<DeleteIcon />}
                          size="sm"
                          variant="ghost"
                          colorScheme="red"
                          onClick={() => handleRemoveContext(index)}
                        />
                      </ListItem>
                    ))}
                  </List>
                )}
              </FormControl>
              
              <FormControl>
                <FormLabel>Image Source</FormLabel>
                <Text fontSize="sm" color="gray.600" p={2} bg="blue.50" borderRadius="md">
                  RAW Mode (always enabled) - All images must have RAW files
                </Text>
              </FormControl>
              
              <FormControl>
                <FormLabel>Preview Resolution</FormLabel>
                <Select
                  value={previewResolution}
                  onChange={(e) => setPreviewResolution(Number(e.target.value))}
                  colorScheme="brand"
                >
                  <option value={450}>450px</option>
                  <option value={1500}>1500px</option>
                  <option value={3000}>3000px</option>
                </Select>
                <Text fontSize="sm" color="gray.600" mt={1}>
                  Resolution for preview images generated from RAW files
                </Text>
              </FormControl>
              
              <FormControl display="flex" alignItems="center">
                <FormLabel mb={0}>Overwrite existing files</FormLabel>
                <Switch
                  isChecked={overwrite}
                  onChange={(e) => setOverwrite(e.target.checked)}
                  colorScheme="brand"
                />
              </FormControl>
              
              <FormControl display="flex" alignItems="center">
                <FormLabel mb={0}>Override number of clusters</FormLabel>
                <Switch
                  isChecked={useCustomK}
                  onChange={(e) => setUseCustomK(e.target.checked)}
                  colorScheme="brand"
                />
              </FormControl>
              
              {useCustomK && (
                <FormControl>
                  <FormLabel>Number of clusters (K)</FormLabel>
                  <NumberInput
                    value={customK}
                    onChange={(_, val) => setCustomK(val)}
                    min={1}
                    max={20}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>
              )}
              
              <FormControl>
                <FormLabel>
                  Clustering Sensitivity: {sensitivity.toFixed(1)}
                </FormLabel>
                <Slider
                  value={sensitivity}
                  onChange={(val) => setSensitivity(val)}
                  min={0.2}
                  max={1.5}
                  step={0.1}
                  colorScheme="brand"
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb />
                </Slider>
                <Text fontSize="sm" color="gray.600" mt={1}>
                  Higher values create more clusters for finer distinctions
                </Text>
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
                  Start Color Correction
                </Button>
              </VStack>
            </TabPanel>
            
            <TabPanel>
              <Card>
                <CardBody>
                  {sessionsData?.persisted_sessions && sessionsData.persisted_sessions.length > 0 ? (
                    <TableContainer>
                      <Table variant="simple">
                        <Thead>
                          <Tr>
                            <Th>Session ID</Th>
                            <Th>Saved At</Th>
                            <Th>Contexts</Th>
                            <Th>Clusters</Th>
                            <Th>Images</Th>
                            <Th>Actions</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {sessionsData.persisted_sessions.map((session: SessionInfo) => (
                            <Tr key={session.session_id}>
                              <Td>
                                <Code fontSize="xs">{session.session_id.slice(0, 8)}...</Code>
                              </Td>
                              <Td>{formatDate(session.saved_at)}</Td>
                              <Td>{session.context_count}</Td>
                              <Td>{session.cluster_count}</Td>
                              <Td>{session.image_count}</Td>
                              <Td>
                                <HStack spacing={2}>
                                  <Button
                                    size="sm"
                                    colorScheme="brand"
                                    onClick={() => restoreMutation.mutate(session.session_id)}
                                    isLoading={restoreMutation.isLoading}
                                  >
                                    Restore
                                  </Button>
                                  <IconButton
                                    aria-label="Delete session"
                                    icon={<DeleteIcon />}
                                    size="sm"
                                    colorScheme="red"
                                    variant="ghost"
                                    onClick={() => deleteMutation.mutate(session.session_id)}
                                    isLoading={deleteMutation.isLoading}
                                  />
                                </HStack>
                              </Td>
                            </Tr>
                          ))}
                        </Tbody>
                      </Table>
                    </TableContainer>
                  ) : (
                    <Text color="gray.600" textAlign="center" py={8}>
                      No saved sessions found
                    </Text>
                  )}
                </CardBody>
              </Card>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </Container>
  )
}

export default SessionSetup

