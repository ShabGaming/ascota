import { useState, useEffect } from 'react'
import {
  Box,
  Button,
  Container,
  Heading,
  VStack,
  HStack,
  SimpleGrid,
  Image,
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
  Text,
} from '@chakra-ui/react'
import { useQuery, useMutation } from 'react-query'
import {
  generateMasks,
  getStage2Results,
  saveStage2,
  getSessionImages,
  getImageUrl,
} from '../api/client'
import { useSessionStore } from '../state/session'
import MaskPainter from '../components/MaskPainter'
import ImageWithMask from '../components/ImageWithMask'

interface Stage2MaskProps {
  sessionId: string
  onComplete: () => void
  onReset: () => void
}

function Stage2Mask({ sessionId, onComplete, onReset }: Stage2MaskProps) {
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null)
  const { isOpen, onOpen, onClose } = useDisclosure()
  const toast = useToast()
  
  const setStage2Results = useSessionStore(state => state.setStage2Results)
  const setStage1Results = useSessionStore(state => state.setStage1Results)
  const setImages = useSessionStore(state => state.setImages)
  const stage2Results = useSessionStore(state => state.stage2Results)
  const images = useSessionStore(state => state.images)
  
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
  
  // Generate masks
  const { isLoading: isGenerating, refetch: refetchGeneration } = useQuery(
    ['generateMasks', sessionId],
    () => generateMasks(sessionId),
    {
      enabled: false,
      onSuccess: (data) => {
        setStage2Results(data.results)
        toast({
          title: 'Mask generation complete',
          status: 'success',
          duration: 3000,
        })
      },
      onError: (error) => {
        toast({
          title: 'Generation failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  // Load existing results for Stage 2
  const { isLoading: isLoadingResults, refetch: refetchStage2Results } = useQuery(
    ['stage2Results', sessionId],
    () => getStage2Results(sessionId),
    {
      onSuccess: (data) => {
        setStage2Results(data.results)
      },
    }
  )
  
  // Load Stage 1 results (needed for mask generation)
  useQuery(
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
    () => saveStage2(sessionId),
    {
      onSuccess: (data) => {
        toast({
          title: 'Saved',
          description: data.message,
          status: 'success',
          duration: 3000,
        })
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
    setSelectedImageId(imageId)
    onOpen()
  }
  
  const handleGenerate = () => {
    refetchGeneration()
  }
  
  const handleNext = () => {
    saveMutation.mutate()
  }
  
  if (isLoadingResults || isLoadingImages) {
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
          <Heading size="lg">Stage 2: Background Segmentation</Heading>
          <HStack>
            <Button onClick={handleGenerate} isLoading={isGenerating}>
              Generate Masks
            </Button>
            <Button onClick={handleNext} colorScheme="brand" isLoading={saveMutation.isLoading}>
              Next Stage
            </Button>
            <Button onClick={onReset} variant="ghost">
              Reset
            </Button>
          </HStack>
        </HStack>
        
        <Text color="gray.600">
          Click on an image to edit the mask. Use the brush tool to paint in (foreground) or paint out (background).
        </Text>
        
        <SimpleGrid columns={{ base: 2, md: 3, lg: 4 }} spacing={4}>
          {imageList.map((image) => {
            const result = stage2Results[image.id]
            
            return (
              <Box
                key={image.id}
                position="relative"
                cursor="pointer"
                onClick={() => handleImageClick(image.id)}
                border="2px"
                borderColor="gray.200"
                borderRadius="md"
                overflow="hidden"
                bg="gray.100"
                _hover={{ borderColor: 'brand.500' }}
              >
                <ImageWithMask
                  key={`${image.id}-${result?.mask_path || 'no-mask'}`}
                  sessionId={sessionId}
                  image={image}
                  result={result}
                  maxHeight="200px"
                  opacity={0.5}
                />
                <Box p={2} bg="white">
                  <HStack justify="space-between">
                    <Text fontSize="xs" fontWeight="bold">
                      {image.find_number}
                    </Text>
                    {result?.error ? (
                      <Badge colorScheme="red">Error</Badge>
                    ) : result?.mask_path ? (
                      <Badge colorScheme="green">Mask Ready</Badge>
                    ) : (
                      <Badge colorScheme="gray">No Mask</Badge>
                    )}
                  </HStack>
                </Box>
              </Box>
            )
          })}
        </SimpleGrid>
        
        <Modal isOpen={isOpen} onClose={onClose} size="full">
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Edit Mask</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              {selectedImageId && (
                <MaskPainter
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

export default Stage2Mask

