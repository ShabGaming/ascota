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
  const [imageSource, setImageSource] = useState<"450px" | "1500px" | "3000px" | "raw_mode">("3000px")
  const [overwrite, setOverwrite] = useState(false)
  const [useCustomK, setUseCustomK] = useState(false)
  const [customK, setCustomK] = useState(3)
  const [sensitivity, setSensitivity] = useState(1.0)
  const [isLoading, setIsLoading] = useState(false)
  
  const toast = useToast()
  const setSession = useSessionStore(state => state.setSession)
  
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
          image_source: imageSource,
          overwrite,
          custom_k: useCustomK ? customK : undefined,
          sensitivity,
        },
      })
      
      setSession(response.session_id)
      
      toast({
        title: 'Session created',
        description: 'Starting clustering...',
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
          <Heading size="xl" mb={2}>Color Correction Tool</Heading>
          <Text color="gray.600">
            Batch color correction with intelligent clustering
          </Text>
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
                <Select
                  value={imageSource}
                  onChange={(e) => setImageSource(e.target.value as typeof imageSource)}
                  colorScheme="brand"
                >
                  <option value="450px">450px</option>
                  <option value="1500px">1500px</option>
                  <option value="3000px">3000px</option>
                  <option value="raw_mode">Raw Mode</option>
                </Select>
                <Text fontSize="sm" color="gray.600" mt={1}>
                  {imageSource === "3000px" && "Uses 3000px, falls back to 1500px, then 450px"}
                  {imageSource === "1500px" && "Uses 1500px, falls back to 450px"}
                  {imageSource === "450px" && "Uses 450px only"}
                  {imageSource === "raw_mode" && "Uses RAW files, falls back to 3000px, 1500px, then 450px"}
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
    </Container>
  )
}

export default SessionSetup

