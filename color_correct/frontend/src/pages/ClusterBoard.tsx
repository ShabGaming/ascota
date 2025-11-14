import { useEffect, useState } from 'react'
import {
  Box,
  Container,
  Heading,
  HStack,
  VStack,
  Button,
  Spinner,
  Text,
  useToast,
  Progress,
  Card,
  CardBody,
  List,
  ListItem,
  Code,
  Divider,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  IconButton,
  FormControl,
  FormLabel,
} from '@chakra-ui/react'
import { ArrowBackIcon } from '@chakra-ui/icons'
import { DragDropContext, DropResult } from 'react-beautiful-dnd'
import { useQuery } from 'react-query'
import {
  startClustering,
  getJobStatus,
  getClusters,
  moveImage,
  getSessionInfo,
  getDiscoveredImages,
  ImageItem,
  createCluster,
  undoClusters,
} from '../api/client'
import { useSessionStore } from '../state/session'
import ClusterColumn from '../components/ClusterColumn'
import CorrectionPanel from '../components/CorrectionPanel'
import ExportBar from '../components/ExportBar'

interface ClusterBoardProps {
  sessionId: string
  onClusteringComplete: () => void
  onReset: () => void
  isLoadingClusters: boolean  // Not used anymore, but kept for compatibility
}

function ClusterBoard({
  sessionId,
  onClusteringComplete,
  onReset,
  isLoadingClusters: initialLoading,
}: ClusterBoardProps) {
  const [clusterJobId, setClusterJobId] = useState<string | null>(null)
  const [isPolling, setIsPolling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState('Initializing...')
  const [showDiscoveredImages, setShowDiscoveredImages] = useState(true)
  const [discoveredImages, setDiscoveredImages] = useState<Record<string, ImageItem> | null>(null)
  const [reclusterSensitivity, setReclusterSensitivity] = useState(1.0)
  const [canUndo, setCanUndo] = useState(false)
  
  const toast = useToast()
  const { clusters, images, setClusters, moveImageLocally, selectedImageId, selectedClusterId } = useSessionStore()
  
  // Get session info and discovered images
  const { data: sessionInfo } = useQuery(
    ['sessionInfo', sessionId],
    () => getSessionInfo(sessionId),
    {
      enabled: true,
      onSuccess: async (data) => {
        // Always fetch discovered images
        try {
          const discovered = await getDiscoveredImages(sessionId)
          setDiscoveredImages(discovered.images)
        } catch (error) {
          toast({
            title: 'Failed to fetch discovered images',
            description: error instanceof Error ? error.message : 'Unknown error',
            status: 'error',
            duration: 5000,
          })
        }
      },
    }
  )
  
  // Poll job status
  useQuery(
    ['jobStatus', sessionId, clusterJobId],
    () => getJobStatus(sessionId, clusterJobId!),
    {
      enabled: isPolling && !!clusterJobId,
      refetchInterval: 1000,
      onSuccess: (data) => {
        setProgress(data.progress * 100)
        setStatusMessage(data.message || 'Processing...')
        
        if (data.status === 'completed') {
          setIsPolling(false)
          onClusteringComplete()
          // Refetch clusters to get updated state
          refetchClusters()
          
          toast({
            title: 'Clustering complete',
            status: 'success',
            duration: 3000,
          })
        } else if (data.status === 'failed') {
          setIsPolling(false)
          toast({
            title: 'Clustering failed',
            description: data.error,
            status: 'error',
            duration: 5000,
          })
        }
      },
    }
  )
  
  // Fetch clusters
  const { isLoading: isFetchingClusters, refetch: refetchClusters } = useQuery(
    ['clusters', sessionId],
    () => getClusters(sessionId),
    {
      enabled: !isPolling,
      onSuccess: (data) => {
        setClusters(data)
      },
    }
  )
  
  // Update sensitivity from session info
  useEffect(() => {
    if (sessionInfo?.options.sensitivity) {
      setReclusterSensitivity(sessionInfo.options.sensitivity)
    }
  }, [sessionInfo])
  
  const handleRecluster = async () => {
    try {
      setIsPolling(true)
      setCanUndo(true)  // Enable undo after reclustering starts
      const response = await startClustering(sessionId, reclusterSensitivity)
      setClusterJobId(response.job_id)
    } catch (error) {
      setIsPolling(false)
      toast({
        title: 'Failed to start reclustering',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      })
    }
  }
  
  const handleUndo = async () => {
    try {
      await undoClusters(sessionId)
      await refetchClusters()
      setCanUndo(false)  // Disable undo after use (one-time only)
      toast({
        title: 'Clusters restored',
        status: 'success',
        duration: 2000,
      })
    } catch (error) {
      toast({
        title: 'Undo failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
      })
      // If undo fails (e.g., no previous state), disable the button
      setCanUndo(false)
    }
  }
  
  const handleCreateCluster = async () => {
    try {
      await createCluster(sessionId)
      // Refetch clusters to show the new one
      await refetchClusters()
      toast({
        title: 'New cluster created',
        status: 'success',
        duration: 2000,
      })
    } catch (error) {
      toast({
        title: 'Failed to create cluster',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
      })
    }
  }
  
  const handleDragEnd = async (result: DropResult) => {
    const { source, destination, draggableId } = result
    
    if (!destination) return
    if (source.droppableId === destination.droppableId) return
    
    const imageId = draggableId
    const targetClusterId = destination.droppableId
    
    // Optimistic update
    moveImageLocally(imageId, targetClusterId)
    
    try {
      await moveImage(sessionId, imageId, targetClusterId)
    } catch (error) {
      toast({
        title: 'Failed to move image',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
      })
      
      // Revert by refetching
      const data = await getClusters(sessionId)
      setClusters(data)
    }
  }
  
  // Show discovered images list before clustering
  if (showDiscoveredImages && discoveredImages) {
    const imageList = Object.values(discoveredImages)
    const formatPath = (img: ImageItem) => {
      const filename = img.primary_path.split(/[/\\]/).pop() || img.primary_path
      return filename
    }
    
    const getSourceLabel = (source: string) => {
      switch (source) {
        case "450px": return "450px (450px only)"
        case "1500px": return "1500px (1500px → 450px)"
        case "3000px": return "3000px (3000px → 1500px → 450px)"
        case "raw_mode": return "Raw Mode (RAW → 3000px → 1500px → 450px)"
        default: return source
      }
    }
    
    return (
      <Container maxW="container.lg" py={10}>
        <VStack spacing={6} align="stretch">
          <Box>
            <Heading size="lg" mb={2}>Discovered Images</Heading>
            <Text color="gray.600" mb={4}>
              Found {imageList.length} images. Review the list below before clustering starts.
            </Text>
          </Box>
          
          <Card>
            <CardBody>
              <VStack align="stretch" spacing={4} maxH="60vh" overflowY="auto">
              <List spacing={2}>
                {imageList.map((img) => (
                  <ListItem key={img.id} p={2} bg="gray.50" borderRadius="md">
                    <HStack justify="space-between">
                      <VStack align="start" spacing={0}>
                        <Text fontWeight="medium" fontSize="sm">
                          Find {img.find_number} · Context: {img.context_id.split(/[/\\]/).pop()}
                        </Text>
                        <Code fontSize="xs" colorScheme="gray">
                          {formatPath(img)}
                        </Code>
                        <HStack spacing={2} mt={1}>
                          {img.proxy_450 && <Text fontSize="xs" color="green.600">450px</Text>}
                          {img.proxy_1500 && <Text fontSize="xs" color="green.600">1500px</Text>}
                          {img.proxy_3000 && <Text fontSize="xs" color="green.600">3000px</Text>}
                          {img.raw_path && <Text fontSize="xs" color="blue.600">RAW</Text>}
                        </HStack>
                      </VStack>
                      <Text fontSize="xs" color="gray.500" fontFamily="mono">
                        {img.id.slice(0, 8)}
                      </Text>
                    </HStack>
                  </ListItem>
                ))}
              </List>
              </VStack>
            </CardBody>
          </Card>
          
          <Divider />
          
          <HStack justify="space-between">
            <Text fontSize="sm" color="gray.600">
              Image source: {sessionInfo?.options.image_source ? getSourceLabel(sessionInfo.options.image_source) : 'Unknown'}
            </Text>
            <Button
              colorScheme="brand"
              onClick={async () => {
                setShowDiscoveredImages(false)
                setIsPolling(true)
                // Start clustering now
                try {
                  const response = await startClustering(sessionId)
                  setClusterJobId(response.job_id)
                } catch (error) {
                  setIsPolling(false)
                  toast({
                    title: 'Failed to start clustering',
                    description: error instanceof Error ? error.message : 'Unknown error',
                    status: 'error',
                    duration: 5000,
                  })
                }
              }}
            >
              Continue to Clustering
            </Button>
          </HStack>
        </VStack>
      </Container>
    )
  }
  
  if (isPolling) {
    return (
      <Container maxW="container.md" py={20}>
        <VStack spacing={6}>
          <Spinner size="xl" color="brand.500" />
          <Heading size="md">{statusMessage}</Heading>
          <Box w="full">
            <Progress value={progress} colorScheme="brand" size="lg" borderRadius="md" />
            <Text textAlign="center" mt={2} color="gray.600">
              {progress.toFixed(0)}%
            </Text>
          </Box>
        </VStack>
      </Container>
    )
  }
  
  if (isFetchingClusters) {
    return (
      <Container maxW="container.md" py={20}>
        <VStack spacing={4}>
          <Spinner size="xl" color="brand.500" />
          <Heading size="md">Loading clusters...</Heading>
        </VStack>
      </Container>
    )
  }
  
  return (
    <Box minH="100vh">
      {/* Header */}
      <Box bg="white" borderBottom="1px" borderColor="gray.200" py={4} position="sticky" top={0} zIndex={10}>
        <Container maxW="full" px={6}>
          <VStack align="stretch" spacing={3}>
            <HStack justify="space-between">
              <VStack align="start" spacing={0}>
                <Heading size="lg">Color Correction</Heading>
                <Text fontSize="sm" color="gray.600">
                  {clusters.length} clusters · {Object.keys(images).length} images
                </Text>
              </VStack>
              <HStack spacing={2}>
                <Button colorScheme="brand" onClick={handleCreateCluster}>
                  New Cluster
                </Button>
                <Button variant="outline" onClick={onReset}>
                  New Session
                </Button>
              </HStack>
            </HStack>
            
            {/* Recluster controls */}
            <HStack spacing={4} align="center">
              <FormControl flex="1" maxW="300px">
                <HStack spacing={2}>
                  <FormLabel fontSize="sm" mb={0} minW="80px">
                    Sensitivity: {reclusterSensitivity.toFixed(1)}
                  </FormLabel>
                  <Slider
                    value={reclusterSensitivity}
                    onChange={(val) => setReclusterSensitivity(val)}
                    min={0.2}
                    max={1.5}
                    step={0.1}
                    colorScheme="brand"
                    flex="1"
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </HStack>
              </FormControl>
              
              <Button
                size="sm"
                colorScheme="brand"
                variant="outline"
                onClick={handleRecluster}
                isDisabled={isPolling}
              >
                Recluster
              </Button>
              
              {canUndo && (
                <IconButton
                  aria-label="Undo reclustering"
                  icon={<ArrowBackIcon />}
                  size="sm"
                  variant="outline"
                  onClick={handleUndo}
                  isDisabled={isPolling}
                />
              )}
            </HStack>
          </VStack>
        </Container>
      </Box>
      
      {/* Main content */}
      <Box>
        <DragDropContext onDragEnd={handleDragEnd}>
          <HStack align="start" spacing={4} p={6} overflowX="auto" h="calc(100vh - 180px)">
            {clusters.map((cluster) => (
              <ClusterColumn
                key={cluster.id}
                cluster={cluster}
                images={images}
                sessionId={sessionId}
              />
            ))}
          </HStack>
        </DragDropContext>
      </Box>
      
      {/* Correction panel (sticky) */}
      {selectedImageId && selectedClusterId && (
        <CorrectionPanel
          sessionId={sessionId}
          clusterId={selectedClusterId}
          imageId={selectedImageId}
        />
      )}
      
      {/* Export bar */}
      <ExportBar sessionId={sessionId} />
    </Box>
  )
}

export default ClusterBoard

