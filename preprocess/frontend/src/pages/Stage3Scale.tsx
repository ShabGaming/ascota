import { useState } from 'react'
import {
  Button,
  Container,
  Heading,
  VStack,
  HStack,
  SimpleGrid,
  Card,
  CardBody,
  Text,
  Badge,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
} from '@chakra-ui/react'
import { useQuery, useMutation } from 'react-query'
import {
  calculateScale,
  getStage3Results,
  saveStage3,
  getSessionImages,
  getStage1Results,
} from '../api/client'
import { useSessionStore } from '../state/session'
import ScaleEditor from '../components/ScaleEditor'
import ImageWithScaleCard from '../components/ImageWithScaleCard'

interface Stage3ScaleProps {
  sessionId: string
  onComplete: () => void
  onReset: () => void
}

function Stage3Scale({ sessionId, onComplete, onReset }: Stage3ScaleProps) {
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null)
  const { isOpen, onOpen, onClose } = useDisclosure()
  const toast = useToast()
  
  const setStage3Results = useSessionStore(state => state.setStage3Results)
  const setStage1Results = useSessionStore(state => state.setStage1Results)
  const setImages = useSessionStore(state => state.setImages)
  const stage3Results = useSessionStore(state => state.stage3Results)
  const images = useSessionStore(state => state.images)
  const stage1Results = useSessionStore(state => state.stage1Results)
  
  // Load images from session
  const { isLoading: isLoadingImages } = useQuery(
    ['sessionImages', sessionId],
    () => getSessionImages(sessionId),
    {
      onSuccess: (data) => {
        setImages(data.images)
      },
    }
  )
  
  // Calculate scale
  const { isLoading: isCalculating, refetch: refetchCalculation } = useQuery(
    ['calculateScale', sessionId],
    () => calculateScale(sessionId),
    {
      enabled: false,
      onSuccess: (data) => {
        setStage3Results(data.results)
        toast({
          title: 'Scale calculation complete',
          status: 'success',
          duration: 3000,
        })
      },
      onError: (error) => {
        toast({
          title: 'Calculation failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  // Load existing results for Stage 3
  const { isLoading: isLoadingResults } = useQuery(
    ['stage3Results', sessionId],
    () => getStage3Results(sessionId),
    {
      onSuccess: (data) => {
        setStage3Results(data.results)
      },
    }
  )
  
  // Load Stage 1 results (needed for card visualization)
  const { isLoading: isLoadingStage1 } = useQuery(
    ['stage1Results', sessionId],
    () => getStage1Results(sessionId),
    {
      onSuccess: (data) => {
        setStage1Results(data.results)
      },
    }
  )
  
  // Save mutation
  const saveMutation = useMutation(
    () => saveStage3(sessionId),
    {
      onSuccess: (data) => {
        toast({
          title: 'Saved',
          description: data.message,
          status: 'success',
          duration: 3000,
        })
        // Navigate back to homepage after saving
        onComplete()
      },
      onError: (error) => {
        toast({
          title: 'Save failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  const handleImageClick = (imageId: string) => {
    const result = stage3Results[imageId]
    
    // Only allow editing for 8-hybrid cards
    if (result?.method === '8_hybrid_card' && !result.error) {
      setSelectedImageId(imageId)
      onOpen()
    }
  }
  
  const handleCalculate = () => {
    refetchCalculation()
  }
  
  const handleSave = () => {
    saveMutation.mutate()
  }
  
  if (isLoadingResults || isLoadingImages || isLoadingStage1) {
    return (
      <Container maxW="container.xl" py={10}>
        <VStack spacing={4}>
          <Spinner size="xl" />
          <Text>Loading images...</Text>
        </VStack>
      </Container>
    )
  }
  
  const imageList = Object.values(images)
  
  if (imageList.length === 0) {
    return (
      <Container maxW="container.xl" py={10}>
        <VStack spacing={4}>
          <Text>No images found in session.</Text>
        </VStack>
      </Container>
    )
  }
  
  return (
    <Container maxW="container.xl" py={10}>
      <VStack spacing={6} align="stretch">
        <HStack justify="space-between">
          <Heading size="lg">Stage 3: Scale Calculation</Heading>
          <HStack>
            <Button onClick={handleCalculate} isLoading={isCalculating}>
              Calculate Scale
            </Button>
            <Button onClick={handleSave} colorScheme="brand" isLoading={saveMutation.isLoading}>
              Save Results
            </Button>
            <Button onClick={onReset} variant="ghost">
              Reset
            </Button>
          </HStack>
        </HStack>
        
        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={4}>
          {imageList.map((image) => {
            const result = stage3Results[image.id]
            const stage1Data = stage1Results[image.id]
            const canEdit = result?.method === '8_hybrid_card' && !result.error
            
            // Find the card used for scale calculation
            const scaleCard = result?.card_used && stage1Data?.cards
              ? stage1Data.cards.find(card => card.card_id === result.card_used)
              : undefined
            
            return (
              <Card
                key={image.id}
                cursor={canEdit ? 'pointer' : 'default'}
                onClick={() => canEdit && handleImageClick(image.id)}
                _hover={canEdit ? { boxShadow: 'lg' } : {}}
              >
                <CardBody>
                  <VStack align="stretch" spacing={2}>
                    {/* Image with card visualization */}
                    {result && scaleCard && (
                      <ImageWithScaleCard
                        sessionId={sessionId}
                        image={image}
                        result={result}
                        card={scaleCard}
                        maxHeight="150px"
                      />
                    )}
                    
                    <HStack justify="space-between">
                      <Text fontWeight="bold">{image.find_number}</Text>
                      {result?.error ? (
                        <Badge colorScheme="red">Error</Badge>
                      ) : result?.pixels_per_cm ? (
                        <Badge colorScheme="green">Complete</Badge>
                      ) : (
                        <Badge colorScheme="gray">Pending</Badge>
                      )}
                    </HStack>
                    
                    {result?.error ? (
                      <Text fontSize="sm" color="red.500">
                        {result.error}
                      </Text>
                    ) : result?.pixels_per_cm ? (
                      <>
                        <Stat size="sm">
                          <StatLabel>Pixels per cm</StatLabel>
                          <StatNumber>{result.pixels_per_cm.toFixed(0)}</StatNumber>
                        </Stat>
                        {result.surface_area_cm2 && (
                          <Stat size="sm">
                            <StatLabel>Surface Area</StatLabel>
                            <StatNumber>{result.surface_area_cm2.toFixed(2)} cm²</StatNumber>
                          </Stat>
                        )}
                        <Text fontSize="xs" color="gray.600">
                          Method: {result.method}
                        </Text>
                        {canEdit && (
                          <Text fontSize="xs" color="blue.500" fontStyle="italic">
                            Click to edit circle centers
                          </Text>
                        )}
                      </>
                    ) : (
                      <Text fontSize="sm" color="gray.500">
                        No scale calculated
                      </Text>
                    )}
                  </VStack>
                </CardBody>
              </Card>
            )
          })}
        </SimpleGrid>
        
        <Modal isOpen={isOpen} onClose={onClose} size="6xl">
          <ModalOverlay />
          <ModalContent maxW="90vw" maxH="90vh">
            <ModalHeader>Edit Circle Centers (8-Hybrid Card)</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              {selectedImageId && (
                <ScaleEditor
                  sessionId={sessionId}
                  imageId={selectedImageId}
                  onClose={onClose}
                />
              )}
            </ModalBody>
          </ModalContent>
        </Modal>
      </VStack>
    </Container>
  )
}

export default Stage3Scale

