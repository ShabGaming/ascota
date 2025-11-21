import { Box, Image, Text } from '@chakra-ui/react'
import { Draggable } from 'react-beautiful-dnd'
import { ImageItem, getPreviewUrl } from '../api/client'
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

function ImageTile({ image, index, sessionId, clusterId, isFullscreen = false, showOverall = true, showIndividual = true }: ImageTileProps) {
  const { 
    selectedImageId, 
    setSelectedImage, 
    clusters,
    pendingOverallCorrection,
    pendingIndividualCorrection
  } = useSessionStore()
  const isSelected = selectedImageId === image.id
  
  const handleClick = () => {
    setSelectedImage(image.id, clusterId)
  }
  
  // Get cluster to access correction params
  const cluster = clusters.find(c => c.id === clusterId)
  
  // Use pending corrections for real-time preview if available, otherwise use saved corrections
  const overallParams = pendingOverallCorrection[clusterId] || cluster?.correction_params
  const individualParams = pendingIndividualCorrection[image.id] || undefined
  
  // Create preview URL - use pending overall for cluster preview, individual for this image
  const previewSize = isFullscreen ? 800 : 400
  // For preview, use overall params (cluster-level) - individual will be applied by backend
  const previewUrl = getPreviewUrl(
    sessionId, 
    image.id, 
    clusterId, 
    previewSize, 
    overallParams,
    individualParams,
    showOverall,
    showIndividual
  )
  
  // Use a key that changes when corrections or visibility change to force image reload
  const imageKey = `${image.id}-${clusterId}-${JSON.stringify(overallParams)}-${JSON.stringify(individualParams)}-${showOverall}-${showIndividual}`
  
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
          <Image
            key={imageKey}
            src={previewUrl}
            alt={`Image ${image.find_number}`}
            w="full"
            h={isFullscreen ? "400px" : "180px"}
            objectFit="cover"
            fallbackSrc="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ddd' width='100' height='100'/%3E%3C/svg%3E"
          />
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
}

export default ImageTile

