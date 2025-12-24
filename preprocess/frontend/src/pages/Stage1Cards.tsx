import { useState, useEffect, useRef } from 'react'
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
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
} from '@chakra-ui/react'
import { useQuery, useMutation } from 'react-query'
import {
  detectCards,
  getStage1Results,
  saveStage1,
  getSessionImages,
  getImageUrl,
  ImageItem,
} from '../api/client'
import { useSessionStore } from '../state/session'
import CardEditor from '../components/CardEditor'
import ImageWithCards from '../components/ImageWithCards'

interface Stage1CardsProps {
  sessionId: string
  onComplete: () => void
  onReset: () => void
}

function Stage1Cards({ sessionId, onComplete, onReset }: Stage1CardsProps) {
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null)
  const { isOpen, onOpen, onClose } = useDisclosure()
  const { isOpen: isConfirmOpen, onOpen: onConfirmOpen, onClose: onConfirmClose } = useDisclosure()
  const cancelRef = useRef<HTMLButtonElement>(null)
  const toast = useToast()
  
  const setStage1Results = useSessionStore(state => state.setStage1Results)
  const setImages = useSessionStore(state => state.setImages)
  const stage1Results = useSessionStore(state => state.stage1Results)
  const images = useSessionStore(state => state.images)
  
  // Load images from session
  const { isLoading: isLoadingImages } = useQuery(
    ['sessionImages', sessionId],
    () => getSessionImages(sessionId),
    {
      onSuccess: (data) => {
        setImages(data.images)
      },
      onError: (error) => {
        toast({
          title: 'Failed to load images',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  // Detect cards on mount
  const { isLoading: isDetecting, refetch: refetchDetection } = useQuery(
    ['detectCards', sessionId],
    () => detectCards(sessionId),
    {
      enabled: false,
      onSuccess: (data) => {
        setStage1Results(data.results)
        toast({
          title: 'Card detection complete',
          status: 'success',
          duration: 3000,
        })
      },
      onError: (error) => {
        toast({
          title: 'Detection failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  // Load existing results
  const { isLoading: isLoadingResults } = useQuery(
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
    () => saveStage1(sessionId),
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
  
  const handleDetect = () => {
    refetchDetection()
  }
  
  // Determine warning status for an image
  const getWarningStatus = (result?: ImageCardResult): 'none' | 'yellow' | 'red' => {
    if (!result || !result.cards || result.cards.length === 0) {
      return 'none'
    }
    
    const cards = result.cards
    const cardTypes = cards.map(c => c.card_type)
    
    // Red warning: Only 24 color card OR 3+ cards detected
    if (cardTypes.length === 1 && cardTypes[0] === '24_color_card') {
      return 'red'
    }
    if (cardTypes.length >= 3) {
      return 'red'
    }
    
    // Yellow warning: Only checker card
    if (cardTypes.length === 1 && cardTypes[0] === 'checker_card') {
      return 'yellow'
    }
    
    return 'none'
  }
  
  // Check if there are any red warnings
  const hasRedWarnings = () => {
    return imageList.some(image => {
      const result = stage1Results[image.id]
      return getWarningStatus(result) === 'red'
    })
  }
  
  const handleNext = () => {
    // Check for red warnings and show confirmation if needed
    if (hasRedWarnings()) {
      onConfirmOpen()
    } else {
      saveMutation.mutate()
    }
  }
  
  const handleConfirmNext = () => {
    onConfirmClose()
    saveMutation.mutate()
  }
  
  const getCardColor = (cardType: string) => {
    switch (cardType) {
      case '24_color_card':
        return 'green'
      case '8_hybrid_card':
        return 'blue'
      case 'checker_card':
        return 'red'
      default:
        return 'gray'
    }
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
          <Text>No images found in session. Please check that your context paths are correct and contain -3000 images.</Text>
        </VStack>
      </Container>
    )
  }
  
  return (
    <Container maxW="container.xl" py={10}>
      <VStack spacing={6} align="stretch">
        <HStack justify="space-between">
          <Heading size="lg">Stage 1: Card Detection</Heading>
          <HStack>
            <Button onClick={handleDetect} isLoading={isDetecting}>
              Detect Cards
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
          Click on an image to edit card detections. Cards are color-coded: Green (24-color), Blue (8-hybrid), Red (checker).
        </Text>
        
         <SimpleGrid columns={{ base: 2, md: 3, lg: 4 }} spacing={4}>
           {imageList.map((image) => {
             const result = stage1Results[image.id]
             const warningStatus = getWarningStatus(result)
             
             // Determine border color based on warning status
             let borderColor = 'gray.200'
             let hoverBorderColor = 'brand.500'
             if (warningStatus === 'yellow') {
               borderColor = 'yellow.400'
               hoverBorderColor = 'yellow.500'
             } else if (warningStatus === 'red') {
               borderColor = 'red.400'
               hoverBorderColor = 'red.500'
             }
             
             return (
               <Box
                 key={image.id}
                 position="relative"
                 cursor="pointer"
                 onClick={() => handleImageClick(image.id)}
                 border="2px"
                 borderColor={borderColor}
                 borderRadius="md"
                 overflow="hidden"
                 bg="gray.100"
                 _hover={{ borderColor: hoverBorderColor }}
               >
                 <ImageWithCards
                   sessionId={sessionId}
                   image={image}
                   result={result}
                   maxHeight="200px"
                 />
                <Box p={2} bg="white">
                  <HStack justify="space-between" mb={1}>
                    <Text fontSize="xs" fontWeight="bold">
                      {image.find_number}
                    </Text>
                    {result?.error ? (
                      <Badge colorScheme="red">Error</Badge>
                    ) : result?.cards && result.cards.length > 0 ? (
                      <Badge colorScheme="green">
                        {result.cards.length} card(s)
                      </Badge>
                    ) : (
                      <Badge colorScheme="gray">No cards</Badge>
                    )}
                  </HStack>
                  {result?.cards && result.cards.length > 0 && (
                    <HStack spacing={1} flexWrap="wrap">
                      {result.cards.map((card) => (
                        <Badge
                          key={card.card_id}
                          colorScheme={getCardColor(card.card_type)}
                          fontSize="xs"
                        >
                          {card.card_type.replace('_', ' ')}
                        </Badge>
                      ))}
                    </HStack>
                  )}
                  {warningStatus === 'yellow' && (
                    <Text fontSize="xs" color="yellow.600" fontWeight="bold" mt={1}>
                      ⚠️ Warning: Only checker card detected
                    </Text>
                  )}
                  {warningStatus === 'red' && (
                    <Text fontSize="xs" color="red.600" fontWeight="bold" mt={1}>
                      ⚠️ Error: {result?.cards?.length === 1 && result.cards[0].card_type === '24_color_card' 
                        ? 'Only 24-color card detected' 
                        : '3 or more cards detected'}
                    </Text>
                  )}
                </Box>
              </Box>
            )
          })}
        </SimpleGrid>
        
        <Modal isOpen={isOpen} onClose={onClose} size="full">
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Edit Card Detections</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              {selectedImageId && (
                <CardEditor
                  sessionId={sessionId}
                  imageId={selectedImageId}
                  onClose={onClose}
                />
              )}
            </ModalBody>
          </ModalContent>
        </Modal>
        
        <AlertDialog
          isOpen={isConfirmOpen}
          leastDestructiveRef={cancelRef}
          onClose={onConfirmClose}
        >
          <AlertDialogOverlay>
            <AlertDialogContent>
              <AlertDialogHeader fontSize="lg" fontWeight="bold">
                Warning: Card Detection Issues
              </AlertDialogHeader>
              
              <AlertDialogBody>
                Some images have card detection issues:
                <VStack align="stretch" mt={3} spacing={2}>
                  {imageList
                    .filter(image => {
                      const result = stage1Results[image.id]
                      return getWarningStatus(result) === 'red'
                    })
                    .map(image => {
                      const result = stage1Results[image.id]
                      const issue = result?.cards?.length === 1 && result.cards[0].card_type === '24_color_card'
                        ? 'Only 24-color card detected'
                        : `${result?.cards?.length || 0} cards detected (3+)`
                      return (
                        <Text key={image.id} fontSize="sm">
                          • <strong>{image.find_number}</strong>: {issue}
                        </Text>
                      )
                    })}
                </VStack>
                <Text mt={3}>
                  Are you sure you want to proceed to the next stage?
                </Text>
              </AlertDialogBody>
              
              <AlertDialogFooter>
                <Button ref={cancelRef} onClick={onConfirmClose}>
                  Cancel
                </Button>
                <Button colorScheme="red" onClick={handleConfirmNext} ml={3}>
                  Proceed Anyway
                </Button>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogOverlay>
        </AlertDialog>
      </VStack>
    </Container>
  )
}

export default Stage1Cards

