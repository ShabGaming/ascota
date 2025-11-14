import { Box, Image, Text } from '@chakra-ui/react'
import { Draggable } from 'react-beautiful-dnd'
import { ImageItem, getPreviewUrl } from '../api/client'
import { useSessionStore } from '../state/session'

interface ImageTileProps {
  image: ImageItem
  index: number
  sessionId: string
  clusterId: string
}

function ImageTile({ image, index, sessionId, clusterId }: ImageTileProps) {
  const { selectedImageId, setSelectedImage, clusters } = useSessionStore()
  const isSelected = selectedImageId === image.id
  
  const handleClick = () => {
    setSelectedImage(image.id, clusterId)
  }
  
  // Get cluster to access correction params
  const cluster = clusters.find(c => c.id === clusterId)
  // Use cluster correction params if available, otherwise undefined (will use defaults)
  const correctionParams = cluster?.correction_params
  
  // Create preview URL with cluster corrections
  // The preview endpoint will use cluster corrections by default, but we can also pass them explicitly
  const previewUrl = getPreviewUrl(sessionId, image.id, clusterId, 400, correctionParams)
  
  // Use a key that changes when cluster corrections change to force image reload
  const imageKey = `${image.id}-${clusterId}-${JSON.stringify(correctionParams)}`
  
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
            h="180px"
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

