import { useMemo, memo, useState, useEffect, useRef } from 'react'
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

const ClusterColumn = memo(function ClusterColumn({ 
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
        
        // Don't invalidate queries - state update is sufficient
        // Images will update automatically through the store state
        
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
        // Invalidate queries to refresh cluster list (cluster deletion requires refetch)
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
        // Don't invalidate queries - individual corrections are stored per-image
        // Images will update automatically when their individual corrections change
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
  
  // Memoize image list to prevent unnecessary re-renders
  const imageList = useMemo(() => {
    return cluster.image_ids
      .map((imageId) => images[imageId])
      .filter((image): image is ImageItem => image !== undefined)
  }, [cluster.image_ids, images])
  
  // Calculate responsive columns based on container width
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [columns, setColumns] = useState(3)
  
  useEffect(() => {
    if (!isFullscreen) return
    
    let resizeObserver: ResizeObserver | null = null
    
    const updateColumns = () => {
      const element = containerRef.current
      if (!element) return
      const width = element.offsetWidth
      // Each image tile is approximately 200px wide (with spacing)
      // Adjust columns based on available width
      if (width < 300) {
        setColumns(1)
      } else if (width < 500) {
        setColumns(2)
      } else if (width < 700) {
        setColumns(3)
      } else if (width < 900) {
        setColumns(4)
      } else {
        setColumns(5)
      }
    }
    
    // Wait for ref to be set, then set up observer
    const timer = setTimeout(() => {
      const element = containerRef.current
      if (element) {
        updateColumns()
        resizeObserver = new ResizeObserver(updateColumns)
        resizeObserver.observe(element)
      }
    }, 0)
    
    return () => {
      clearTimeout(timer)
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
    }
  }, [isFullscreen, imageList.length])
  
  return (
    <Box
      minW={isFullscreen ? "300px" : "300px"}
      maxW={isFullscreen ? "100%" : "300px"}
      w={isFullscreen ? "100%" : undefined}
      h={isFullscreen ? "100%" : undefined}
      bg="white"
      borderRadius="lg"
      boxShadow="md"
      p={4}
      flexShrink={0}
      flex={isFullscreen ? 1 : undefined}
      display={isFullscreen ? "flex" : undefined}
      flexDirection={isFullscreen ? "column" : undefined}
    >
      <VStack align="stretch" spacing={3} flex={isFullscreen ? 1 : undefined} minH={0}>
        <HStack justify="space-between" align="start" flexShrink={0}>
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
        <HStack spacing={2} flexShrink={0}>
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
              ref={(node) => {
                provided.innerRef(node)
                containerRef.current = node
              }}
              {...provided.droppableProps}
              minH={isFullscreen ? "200px" : "400px"}
              maxH={isFullscreen ? "none" : "calc(100vh - 350px)"}
              flex={isFullscreen ? 1 : undefined}
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
                <SimpleGrid columns={columns} spacing={4}>
                  {imageList.map((image, index) => {
                    const originalIndex = cluster.image_ids.indexOf(image.id)
                    return (
                      <ImageTile
                        key={image.id}
                        image={image}
                        index={originalIndex >= 0 ? originalIndex : index}
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
                  {imageList.map((image, index) => {
                    const originalIndex = cluster.image_ids.indexOf(image.id)
                    return (
                      <ImageTile
                        key={image.id}
                        image={image}
                        index={originalIndex >= 0 ? originalIndex : index}
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
}, (prevProps, nextProps) => {
  // Custom comparison function for React.memo
  // Only re-render if cluster data or props actually change
  return (
    prevProps.cluster.id === nextProps.cluster.id &&
    prevProps.cluster.image_ids.length === nextProps.cluster.image_ids.length &&
    prevProps.cluster.image_ids.every((id, i) => id === nextProps.cluster.image_ids[i]) &&
    JSON.stringify(prevProps.cluster.correction_params) === JSON.stringify(nextProps.cluster.correction_params) &&
    prevProps.sessionId === nextProps.sessionId &&
    prevProps.isFullscreen === nextProps.isFullscreen &&
    prevProps.images === nextProps.images
  )
})

export default ClusterColumn

