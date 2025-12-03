import { useState, useEffect, useRef } from 'react'
import {
  Box,
  VStack,
  HStack,
  Heading,
  FormControl,
  FormLabel,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Button,
  Image,
  Text,
  IconButton,
  useToast,
  Select,
} from '@chakra-ui/react'
import { CloseIcon } from '@chakra-ui/icons'
import { useMutation, useQueryClient } from 'react-query'
import { CorrectionParams, setClusterCorrection, getPreviewUrl, autoCorrectCluster, setIndividualCorrection } from '../api/client'
import { useSessionStore } from '../state/session'

interface CorrectionPanelProps {
  sessionId: string
  clusterId: string
  imageId: string
}

const defaultParams: CorrectionParams = {
  temperature: 0,
  tint: 0,
  exposure: 0,
  contrast: 1.0,
  saturation: 1.0,
  red_gain: 1.0,
  green_gain: 1.0,
  blue_gain: 1.0,
}

// Preset color correction templates
const colorPresets: Record<string, CorrectionParams> = {
  'neutral': {
    temperature: 0,
    tint: 0,
    exposure: 0,
    contrast: 1.0,
    saturation: 1.0,
    red_gain: 1.0,
    green_gain: 1.0,
    blue_gain: 1.0,
  },
  'warm': {
    temperature: 30,
    tint: 5,
    exposure: 0,
    contrast: 1.1,
    saturation: 1.1,
    red_gain: 1.05,
    green_gain: 1.0,
    blue_gain: 0.95,
  },
  'cool': {
    temperature: -30,
    tint: -5,
    exposure: 0,
    contrast: 1.1,
    saturation: 1.0,
    red_gain: 0.95,
    green_gain: 1.0,
    blue_gain: 1.05,
  },
  'high_contrast': {
    temperature: 0,
    tint: 0,
    exposure: 0,
    contrast: 1.3,
    saturation: 1.1,
    red_gain: 1.0,
    green_gain: 1.0,
    blue_gain: 1.0,
  },
  'vibrant': {
    temperature: 10,
    tint: 0,
    exposure: 0.2,
    contrast: 1.2,
    saturation: 1.4,
    red_gain: 1.0,
    green_gain: 1.0,
    blue_gain: 1.0,
  },
  'vintage': {
    temperature: 20,
    tint: 10,
    exposure: 0.1,
    contrast: 1.15,
    saturation: 0.85,
    red_gain: 1.1,
    green_gain: 1.0,
    blue_gain: 0.9,
  },
  'dramatic': {
    temperature: 0,
    tint: 0,
    exposure: -0.2,
    contrast: 1.4,
    saturation: 1.2,
    red_gain: 1.0,
    green_gain: 1.0,
    blue_gain: 1.0,
  },
  'soft': {
    temperature: 5,
    tint: 0,
    exposure: 0.1,
    contrast: 0.9,
    saturation: 0.9,
    red_gain: 1.0,
    green_gain: 1.0,
    blue_gain: 1.0,
  },
  'desaturated': {
    temperature: 0,
    tint: 0,
    exposure: 0,
    contrast: 1.1,
    saturation: 0.5,
    red_gain: 1.0,
    green_gain: 1.0,
    blue_gain: 1.0,
  },
}

const presetLabels: Record<string, string> = {
  'neutral': 'Neutral',
  'warm': 'Warm',
  'cool': 'Cool',
  'high_contrast': 'High Contrast',
  'vibrant': 'Vibrant',
  'vintage': 'Vintage',
  'dramatic': 'Dramatic',
  'soft': 'Soft',
  'desaturated': 'Desaturated',
}

function CorrectionPanel({ sessionId, clusterId, imageId }: CorrectionPanelProps) {
  const toast = useToast()
  const queryClient = useQueryClient()
  const { 
    clusters, 
    updateClusterCorrection, 
    setSelectedImage,
    setPendingOverallCorrection,
    setPendingIndividualCorrection,
    clearPendingCorrections
  } = useSessionStore()
  
  const cluster = clusters.find(c => c.id === clusterId)
  const [layerMode, setLayerMode] = useState<'overall' | 'individual'>('overall')
  const [selectedPreset, setSelectedPreset] = useState<string>('')
  const containerRef = useRef<HTMLDivElement>(null)
  const [isWide, setIsWide] = useState(false)
  
  // Detect container width to determine layout
  useEffect(() => {
    if (!containerRef.current) return
    
    const updateLayout = () => {
      const width = containerRef.current?.offsetWidth || 0
      // Use horizontal layout (preview left, params right) when width >= 700px
      setIsWide(width >= 700)
    }
    
    updateLayout()
    const resizeObserver = new ResizeObserver(updateLayout)
    resizeObserver.observe(containerRef.current)
    
    return () => {
      resizeObserver.disconnect()
    }
  }, [])
  
  // Get initial params based on layer mode
  const getInitialParams = async () => {
    if (layerMode === 'overall') {
      return cluster?.correction_params || defaultParams
    } else {
      // For individual, try to fetch existing individual correction from backend
      // For now, start with defaults - individual corrections will be loaded from session
      // when the panel opens (they're stored in session state)
      return defaultParams
    }
  }
  
  const [params, setParams] = useState<CorrectionParams>(defaultParams)
  const [isLoadingParams, setIsLoadingParams] = useState(true)
  
  useEffect(() => {
    // Load initial params when panel opens or mode changes
    const loadParams = async () => {
      setIsLoadingParams(true)
      if (layerMode === 'overall') {
        setParams(cluster?.correction_params || defaultParams)
      } else {
        // For individual mode, we need to fetch from backend
        // But for now, start with defaults - the backend will use saved individual corrections
        // when generating preview
        setParams(defaultParams)
      }
      setIsLoadingParams(false)
      // Clear pending corrections when switching modes or images
      clearPendingCorrections()
    }
    loadParams()
  }, [clusterId, layerMode, imageId, cluster])
  
  // Update pending corrections in real-time for preview
  useEffect(() => {
    if (layerMode === 'overall') {
      setPendingOverallCorrection(clusterId, params)
    } else {
      setPendingIndividualCorrection(imageId, params)
    }
  }, [params, layerMode, clusterId, imageId])
  
  // Update preview URL whenever params change (real-time preview)
  // Pass params as overall or individual based on layer mode
  const previewUrl = layerMode === 'overall' 
    ? getPreviewUrl(sessionId, imageId, clusterId, 600, params, undefined, true, true)
    : getPreviewUrl(sessionId, imageId, clusterId, 600, undefined, params, true, true)
  
  const applyMutation = useMutation(
    () => {
      if (layerMode === 'overall') {
        return setClusterCorrection(sessionId, clusterId, params)
      } else {
        return setIndividualCorrection(sessionId, imageId, params)
      }
    },
    {
      onSuccess: () => {
        if (layerMode === 'overall' && cluster) {
          updateClusterCorrection(clusterId, {
            ...cluster,
            correction_params: params
          })
        }
        
        // Clear pending corrections after successful apply
        clearPendingCorrections()
        
        // Don't invalidate queries - the state update is already handled by updateClusterCorrection
        // Images will update automatically through the store state
        
        toast({
          title: layerMode === 'overall' ? 'Correction applied to cluster' : 'Individual correction applied',
          description: layerMode === 'overall' 
            ? 'All images in the cluster have been updated'
            : 'Correction applied to this image',
          status: 'success',
          duration: 3000,
        })
      },
      onError: (error) => {
        toast({
          title: 'Failed to apply correction',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 3000,
        })
      },
    }
  )
  
  const whiteBalanceMutation = useMutation(
    () => autoCorrectCluster(sessionId, clusterId, imageId),
    {
      onSuccess: (data) => {
        // Merge white balance gains into current params (stack on top)
        // Keep all other adjustments (temperature, tint, exposure, etc.)
        setParams({
          ...params,
          red_gain: data.params.red_gain,
          green_gain: data.params.green_gain,
          blue_gain: data.params.blue_gain,
        })
        
        toast({
          title: 'White balance applied',
          description: 'RGB gains updated. Adjust other settings as needed.',
          status: 'success',
          duration: 2000,
        })
      },
      onError: (error) => {
        toast({
          title: 'White balance failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 3000,
        })
      },
    }
  )
  
  const handleReset = () => {
    setParams(defaultParams)
    setSelectedPreset('')
  }
  
  const handlePresetChange = (presetKey: string) => {
    setSelectedPreset(presetKey)
    if (presetKey && presetKey !== '' && colorPresets[presetKey]) {
      setParams(colorPresets[presetKey])
    }
  }
  
  const handleClose = () => {
    // Clear pending corrections when closing
    clearPendingCorrections()
    setSelectedImage(null, null)
  }
  
  // Render parameters section
  const renderParameters = () => (
    <>
      <FormControl>
        <FormLabel fontSize="sm">Layer Mode</FormLabel>
        <Select
          value={layerMode}
          onChange={(e) => setLayerMode(e.target.value as 'overall' | 'individual')}
          size="sm"
        >
          <option value="overall">Overall (Cluster)</option>
          <option value="individual">Individual (Image)</option>
        </Select>
      </FormControl>
      
      <FormControl>
        <FormLabel fontSize="sm">Color Presets</FormLabel>
        <Select
          value={selectedPreset}
          onChange={(e) => handlePresetChange(e.target.value)}
          size="sm"
          placeholder="Select a preset..."
        >
          {Object.entries(presetLabels).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </Select>
      </FormControl>
      
      {/* Two-column layout for parameters */}
      <HStack align="start" spacing={4}>
        {/* Left column: Basic parameters */}
        <VStack align="stretch" spacing={4} flex={1}>
          <FormControl>
            <FormLabel fontSize="sm">Temperature: {params.temperature.toFixed(0)}</FormLabel>
            <Slider
              value={params.temperature}
              onChange={(val) => setParams({ ...params, temperature: val })}
              min={-100}
              max={100}
              step={1}
              colorScheme="orange"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
          
          <FormControl>
            <FormLabel fontSize="sm">Tint: {params.tint.toFixed(0)}</FormLabel>
            <Slider
              value={params.tint}
              onChange={(val) => setParams({ ...params, tint: val })}
              min={-100}
              max={100}
              step={1}
              colorScheme="pink"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
          
          <FormControl>
            <FormLabel fontSize="sm">Exposure: {params.exposure.toFixed(2)} EV</FormLabel>
            <Slider
              value={params.exposure}
              onChange={(val) => setParams({ ...params, exposure: val })}
              min={-2}
              max={2}
              step={0.1}
              colorScheme="yellow"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
          
          <FormControl>
            <FormLabel fontSize="sm">Contrast: {params.contrast.toFixed(2)}</FormLabel>
            <Slider
              value={params.contrast}
              onChange={(val) => setParams({ ...params, contrast: val })}
              min={0.5}
              max={2.0}
              step={0.05}
              colorScheme="gray"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
          
          <FormControl>
            <FormLabel fontSize="sm">Saturation: {params.saturation.toFixed(2)}</FormLabel>
            <Slider
              value={params.saturation}
              onChange={(val) => setParams({ ...params, saturation: val })}
              min={0}
              max={2.0}
              step={0.05}
              colorScheme="purple"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
        </VStack>
        
        {/* Right column: RGB Gains */}
        <VStack align="stretch" spacing={4} flex={1}>
          <Text fontSize="sm" fontWeight="bold" mt={0}>RGB Gains</Text>
          
          <FormControl>
            <FormLabel fontSize="sm">Red: {params.red_gain.toFixed(2)}</FormLabel>
            <Slider
              value={params.red_gain}
              onChange={(val) => setParams({ ...params, red_gain: val })}
              min={0.5}
              max={2.0}
              step={0.05}
              colorScheme="red"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
          
          <FormControl>
            <FormLabel fontSize="sm">Green: {params.green_gain.toFixed(2)}</FormLabel>
            <Slider
              value={params.green_gain}
              onChange={(val) => setParams({ ...params, green_gain: val })}
              min={0.5}
              max={2.0}
              step={0.05}
              colorScheme="green"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
          
          <FormControl>
            <FormLabel fontSize="sm">Blue: {params.blue_gain.toFixed(2)}</FormLabel>
            <Slider
              value={params.blue_gain}
              onChange={(val) => setParams({ ...params, blue_gain: val })}
              min={0.5}
              max={2.0}
              step={0.05}
              colorScheme="blue"
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </FormControl>
        </VStack>
      </HStack>
      
      <HStack spacing={2}>
        <Button
          flex={1}
          colorScheme="brand"
          variant="outline"
          onClick={() => whiteBalanceMutation.mutate()}
          isLoading={whiteBalanceMutation.isLoading}
        >
          White Balance
        </Button>
        <Button flex={1} variant="outline" onClick={handleReset}>
          Reset
        </Button>
      </HStack>
      
      <Button
        colorScheme="brand"
        onClick={() => applyMutation.mutate()}
        isLoading={applyMutation.isLoading}
      >
        {layerMode === 'overall' ? 'Apply to Cluster' : 'Apply to Image'}
      </Button>
    </>
  )
  
  // Render preview image
  const renderPreview = () => (
    <Box flexShrink={0}>
      <Image
        key={previewUrl}
        src={previewUrl}
        alt="Preview"
        w="full"
        borderRadius="md"
        boxShadow="md"
      />
    </Box>
  )

  return (
    <Box
      ref={containerRef}
      w="full"
      h="full"
      bg="white"
      display="flex"
      flexDirection="column"
    >
      {isWide ? (
        // Wide layout: Preview on left, parameters on right
        <HStack align="start" spacing={4} h="full" p={6} overflow="hidden">
          {/* Left: Preview */}
          <Box flex="0 0 45%" overflowY="auto" h="full">
            <VStack align="stretch" spacing={4}>
              <HStack justify="space-between">
                <Heading size="md">Correction Panel</Heading>
                <IconButton
                  aria-label="Close panel"
                  icon={<CloseIcon />}
                  size="sm"
                  variant="ghost"
                  onClick={handleClose}
                />
              </HStack>
              {renderPreview()}
            </VStack>
          </Box>
          
          {/* Right: Parameters */}
          <Box flex="1" overflowY="auto" h="full">
            <VStack align="stretch" spacing={4}>
              {renderParameters()}
            </VStack>
          </Box>
        </HStack>
      ) : (
        // Narrow layout: Parameters on top, preview below (current layout)
        <VStack align="stretch" spacing={4} p={6} flex={1} overflowY="auto">
          <HStack justify="space-between">
            <Heading size="md">Correction Panel</Heading>
            <IconButton
              aria-label="Close panel"
              icon={<CloseIcon />}
              size="sm"
              variant="ghost"
              onClick={handleClose}
            />
          </HStack>
          {renderParameters()}
          {renderPreview()}
        </VStack>
      )}
    </Box>
  )
}

export default CorrectionPanel

