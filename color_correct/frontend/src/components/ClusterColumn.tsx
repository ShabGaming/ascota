import {
  Box,
  VStack,
  Heading,
  Badge,
  Button,
  useToast,
  Text,
  HStack,
  IconButton,
} from '@chakra-ui/react'
import { DeleteIcon } from '@chakra-ui/icons'
import { Droppable } from 'react-beautiful-dnd'
import { useMutation, useQueryClient } from 'react-query'
import { Cluster, ImageItem, resetClusterCorrection, deleteCluster } from '../api/client'
import ImageTile from './ImageTile'
import { useSessionStore } from '../state/session'

interface ClusterColumnProps {
  cluster: Cluster
  images: Record<string, ImageItem>
  sessionId: string
}

function ClusterColumn({ cluster, images, sessionId }: ClusterColumnProps) {
  const toast = useToast()
  const queryClient = useQueryClient()
  const updateClusterCorrection = useSessionStore(state => state.updateClusterCorrection)
  
  const resetMutation = useMutation(
    () => resetClusterCorrection(sessionId, cluster.id),
    {
      onSuccess: (data) => {
        const updatedCluster = {
          ...cluster,
          correction_params: data.params
        }
        updateClusterCorrection(cluster.id, updatedCluster)
        
        // Invalidate queries to refresh all image previews
        queryClient.invalidateQueries(['clusters', sessionId])
        
        toast({
          title: 'Correction reset',
          status: 'success',
          duration: 2000,
        })
      },
      onError: (error) => {
        toast({
          title: 'Reset failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 3000,
        })
      },
    }
  )
  
  const deleteMutation = useMutation(
    () => deleteCluster(sessionId, cluster.id),
    {
      onSuccess: () => {
        // Invalidate queries to refresh cluster list
        queryClient.invalidateQueries(['clusters', sessionId])
        
        toast({
          title: 'Cluster deleted',
          status: 'success',
          duration: 2000,
        })
      },
      onError: (error) => {
        toast({
          title: 'Delete failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 3000,
        })
      },
    }
  )
  
  const isEmpty = cluster.image_ids.length === 0
  
  return (
    <Box
      minW="300px"
      maxW="300px"
      bg="white"
      borderRadius="lg"
      boxShadow="md"
      p={4}
      flexShrink={0}
    >
      <VStack align="stretch" spacing={3}>
        <HStack justify="space-between" align="start">
          <Box>
            <Heading size="sm" mb={1}>
              Cluster {cluster.id.slice(0, 8)}
            </Heading>
            <Badge colorScheme="brand">{cluster.image_ids.length} images</Badge>
          </Box>
          {isEmpty && (
            <IconButton
              aria-label="Delete cluster"
              icon={<DeleteIcon />}
              size="xs"
              variant="ghost"
              colorScheme="red"
              onClick={() => deleteMutation.mutate()}
              isLoading={deleteMutation.isLoading}
            />
          )}
        </HStack>
        
        <Button
          size="sm"
          variant="outline"
          onClick={() => resetMutation.mutate()}
          isLoading={resetMutation.isLoading}
        >
          Reset
        </Button>
        
        <Droppable droppableId={cluster.id}>
          {(provided, snapshot) => (
            <Box
              ref={provided.innerRef}
              {...provided.droppableProps}
              minH="400px"
              maxH="calc(100vh - 350px)"
              overflowY="auto"
              bg={snapshot.isDraggingOver ? 'brand.50' : 'gray.50'}
              borderRadius="md"
              p={2}
              transition="background 0.2s"
            >
              <VStack spacing={2}>
                {cluster.image_ids.length === 0 ? (
                  <Box
                    p={8}
                    textAlign="center"
                    color="gray.400"
                    fontSize="sm"
                  >
                    <Text>Empty cluster</Text>
                    <Text fontSize="xs" mt={1}>Drag images here</Text>
                  </Box>
                ) : (
                  cluster.image_ids.map((imageId, index) => {
                    const image = images[imageId]
                    if (!image) return null
                    
                    return (
                      <ImageTile
                        key={imageId}
                        image={image}
                        index={index}
                        sessionId={sessionId}
                        clusterId={cluster.id}
                      />
                    )
                  })
                )}
                {provided.placeholder}
              </VStack>
            </Box>
          )}
        </Droppable>
      </VStack>
    </Box>
  )
}

export default ClusterColumn

