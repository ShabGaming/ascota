import { useMemo, memo, useState, useEffect } from 'react'
import { Box, Image, Text, Skeleton } from '@chakra-ui/react'
import { Draggable } from 'react-beautiful-dnd'
import { ImageItem, getPreviewUrl, hashCorrectionParams } from '../api/client'
import { useSessionStore } from '../state/session'

interface ImageTileProps {
  image: ImageItem
  index: number
  sessionId: string
  clusterId: string
  isFullscreen?: boolean
  showOverall?: boolean
  showIndividual?: boolean
}

const ImageTile = memo(function ImageTile({ 
  image, 
  index, 
  sessionId, 
  clusterId, 
  isFullscreen = false, 
  showOverall = true, 
  showIndividual = true 
}: ImageTileProps) {
  // Use selectors to prevent unnecessary re-renders
  const selectedImageId = useSessionStore(state => state.selectedImageId)
  const setSelectedImage = useSessionStore(state => state.setSelectedImage)
  const cluster = useSessionStore(state => state.clusters.find(c => c.id === clusterId))
  const pendingOverallCorrection = useSessionStore(state => state.pendingOverallCorrection[clusterId])
  const pendingIndividualCorrection = useSessionStore(state => state.pendingIndividualCorrection[image.id])
  
  const isSelected = selectedImageId === image.id
  
  const handleClick = () => {
    // Only allow selection in fullscreen mode
    if (isFullscreen) {
      setSelectedImage(image.id, clusterId)
    }
  }
  
  // Use pending corrections for real-time preview if available, otherwise use saved corrections
  const overallParams = pendingOverallCorrection || cluster?.correction_params
  const individualParams = pendingIndividualCorrection || undefined
  
  // Use consistent preview size to prevent reloads when toggling fullscreen
  // The backend will handle resizing, so we use a fixed size that works for both views
  const previewSize = 600
  
  // Memoize preview URL - only regenerate when params actually change
  const previewUrl = useMemo(() => {
    return getPreviewUrl(
      sessionId, 
      image.id, 
      clusterId, 
      previewSize, 
      overallParams,
      individualParams,
      showOverall,
      showIndividual
    )
  }, [sessionId, image.id, clusterId, previewSize, overallParams, individualParams, showOverall, showIndividual])
  
  // Memoize image key using stable hash function - only changes when corrections actually change
  const imageKey = useMemo(() => {
    const overallHash = hashCorrectionParams(overallParams)
    const individualHash = hashCorrectionParams(individualParams)
    return `${image.id}-${clusterId}-${overallHash}-${individualHash}-${showOverall}-${showIndividual}`
  }, [image.id, clusterId, overallParams, individualParams, showOverall, showIndividual])
  
  // Track image loading state to prevent grey flash
  const [isLoading, setIsLoading] = useState(true)
  const [imageError, setImageError] = useState(false)
  
  // Reset loading state when image key changes (new image URL)
  useEffect(() => {
    setIsLoading(true)
    setImageError(false)
  }, [imageKey])
  
  const handleImageLoad = () => {
    setIsLoading(false)
  }
  
  const handleImageError = () => {
    setIsLoading(false)
    setImageError(true)
  }
  
  return (
    <Draggable draggableId={image.id} index={index}>
      {(provided, snapshot) => (
        <Box
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          w="full"
          cursor="pointer"
          onClick={handleClick}
          borderWidth={isSelected ? '3px' : '2px'}
          borderColor={isSelected ? 'brand.500' : 'gray.200'}
          borderRadius="md"
          overflow="hidden"
          bg="white"
          boxShadow={snapshot.isDragging ? 'xl' : 'sm'}
          transition="all 0.2s"
          _hover={{
            borderColor: 'brand.300',
            transform: 'translateY(-2px)',
            boxShadow: 'md',
          }}
        >
          <Box 
            position="relative" 
            w="full" 
            h={isFullscreen ? undefined : "180px"}
            bg="gray.100"
            overflow="hidden"
            style={isFullscreen ? { aspectRatio: '16/9' } : undefined}
          >
            {isLoading && !imageError && (
              <Skeleton
                position="absolute"
                top={0}
                left={0}
                w="full"
                h="full"
                borderRadius="md"
              />
            )}
            <Image
              key={imageKey}
              src={previewUrl}
              alt={`Image ${image.find_number}`}
              w="full"
              h="full"
              objectFit="cover"
              loading="lazy"
              fallbackSrc="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ddd' width='100' height='100'/%3E%3C/svg%3E"
              opacity={isLoading ? 0 : 1}
              transition="opacity 0.3s ease-in-out"
              onLoad={handleImageLoad}
              onError={handleImageError}
            />
          </Box>
          <Box p={2} bg="gray.50">
            <Text fontSize="xs" fontWeight="medium" noOfLines={1}>
              Find {image.find_number}
            </Text>
            <Text fontSize="xs" color="gray.600" noOfLines={1}>
              Context ...{image.context_id.slice(-20)}
            </Text>
          </Box>
        </Box>
      )}
    </Draggable>
  )
}, (prevProps, nextProps) => {
  // Custom comparison function for React.memo
  // Only re-render if props actually change
  return (
    prevProps.image.id === nextProps.image.id &&
    prevProps.index === nextProps.index &&
    prevProps.sessionId === nextProps.sessionId &&
    prevProps.clusterId === nextProps.clusterId &&
    prevProps.isFullscreen === nextProps.isFullscreen &&
    prevProps.showOverall === nextProps.showOverall &&
    prevProps.showIndividual === nextProps.showIndividual
  )
})

export default ImageTile

