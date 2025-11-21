import {
  Box,
  VStack,
  Heading,
  Badge,
  useToast,
  Text,
  HStack,
  IconButton,
  Button,
  SimpleGrid,
} from '@chakra-ui/react'
import { DeleteIcon, ViewIcon, ViewOffIcon, RepeatIcon } from '@chakra-ui/icons'
import { Droppable } from 'react-beautiful-dnd'
import { useMutation, useQueryClient } from 'react-query'
import { Cluster, ImageItem, resetClusterCorrection, deleteCluster, resetIndividualCorrections } from '../api/client'
import ImageTile from './ImageTile'
import { useSessionStore } from '../state/session'

interface ClusterColumnProps {
  cluster: Cluster
  images: Record<string, ImageItem>
  sessionId: string
  isFullscreen?: boolean
  onToggleFullscreen?: () => void
}

function ClusterColumn({ 
  cluster, 
  images, 
  sessionId, 
  isFullscreen = false, 
  onToggleFullscreen
}: ClusterColumnProps) {
  const toast = useToast()
  const queryClient = useQueryClient()
  const updateClusterCorrection = useSessionStore(state => state.updateClusterCorrection)
  
  // Always show both layers (toggles removed)
  const showOverall = true
  const showIndividual = true
  
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
  
  const resetIndividualMutation = useMutation(
    () => resetIndividualCorrections(sessionId, cluster.id),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['clusters', sessionId])
        toast({
          title: 'Individual corrections reset',
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
  
  const isEmpty = cluster.image_ids.length === 0
  const hasOverallCorrection = cluster.correction_params !== null && cluster.correction_params !== undefined
  
  return (
    <Box
      minW={isFullscreen ? "1200px" : "300px"}
      maxW={isFullscreen ? "100%" : "300px"}
      w={isFullscreen ? "100%" : undefined}
      bg="white"
      borderRadius="lg"
      boxShadow="md"
      p={4}
      flexShrink={0}
      flex={isFullscreen ? 1 : undefined}
    >
      <VStack align="stretch" spacing={3}>
        <HStack justify="space-between" align="start">
          <Box>
            <Heading size="sm" mb={1}>
              Cluster {cluster.id.slice(0, 8)}
            </Heading>
            <Badge colorScheme="brand">{cluster.image_ids.length} images</Badge>
          </Box>
          <HStack spacing={2}>
            {onToggleFullscreen && (
              <IconButton
                aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                icon={isFullscreen ? <ViewOffIcon /> : <ViewIcon />}
                size="xs"
                variant="ghost"
                onClick={onToggleFullscreen}
              />
            )}
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
        </HStack>
        
        {/* Reset Buttons */}
        <HStack spacing={2}>
          <Button
            leftIcon={<RepeatIcon />}
            size="sm"
            variant="outline"
            onClick={() => resetMutation.mutate()}
            isLoading={resetMutation.isLoading}
            isDisabled={!hasOverallCorrection}
            flex={1}
          >
            Reset Overall
          </Button>
          <Button
            leftIcon={<RepeatIcon />}
            size="sm"
            variant="outline"
            onClick={() => resetIndividualMutation.mutate()}
            isLoading={resetIndividualMutation.isLoading}
            flex={1}
          >
            Reset Individual
          </Button>
        </HStack>
        
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
              ) : isFullscreen ? (
                <SimpleGrid columns={3} spacing={4}>
                  {cluster.image_ids.map((imageId, index) => {
                    const image = images[imageId]
                    if (!image) return null
                    
                    return (
                      <ImageTile
                        key={imageId}
                        image={image}
                        index={index}
                        sessionId={sessionId}
                        clusterId={cluster.id}
                        isFullscreen={isFullscreen}
                        showOverall={showOverall}
                        showIndividual={showIndividual}
                      />
                    )
                  })}
                  {provided.placeholder}
                </SimpleGrid>
              ) : (
                <VStack spacing={2}>
                  {cluster.image_ids.map((imageId, index) => {
                    const image = images[imageId]
                    if (!image) return null
                    
                    return (
                      <ImageTile
                        key={imageId}
                        image={image}
                        index={index}
                        sessionId={sessionId}
                        clusterId={cluster.id}
                        isFullscreen={isFullscreen}
                        showOverall={showOverall}
                        showIndividual={showIndividual}
                      />
                    )
                  })}
                  {provided.placeholder}
                </VStack>
              )}
            </Box>
          )}
        </Droppable>
      </VStack>
    </Box>
  )
}

export default ClusterColumn

