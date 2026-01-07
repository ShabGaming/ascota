import { useState, useEffect } from 'react'
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Image,
  IconButton,
  Input,
  Button,
  useToast,
  SimpleGrid,
  Skeleton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from '@chakra-ui/react'
import { CloseIcon, AddIcon, ChevronDownIcon } from '@chakra-ui/icons'

interface ReferencePanelProps {
  sessionId: string
  onClose?: () => void
}

interface ReferenceImage {
  id: string
  name: string
  url: string
  path: string
}

interface PresetReference {
  name: string
  path: string
  image_count: number
}

function ReferencePanel({ sessionId, onClose }: ReferencePanelProps) {
  const toast = useToast()
  const [referenceImages, setReferenceImages] = useState<ReferenceImage[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [presets, setPresets] = useState<PresetReference[]>([])
  const [isLoadingPresets, setIsLoadingPresets] = useState(false)
  const [isLoadingPreset, setIsLoadingPreset] = useState(false)

  // Load reference images and presets on mount
  useEffect(() => {
    loadReferenceImages()
    loadPresets()
  }, [sessionId])

  const loadReferenceImages = async () => {
    try {
      setIsLoading(true)
      const response = await fetch(`/api/sessions/${sessionId}/reference-images`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      if (response.ok) {
        const data = await response.json()
        setReferenceImages(data.images || [])
      } else {
        // If 404 or other error, just set empty array
        setReferenceImages([])
      }
    } catch (error) {
      console.error('Failed to load reference images:', error)
      setReferenceImages([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    setIsUploading(true)
    try {
      const formData = new FormData()
      Array.from(files).forEach(file => {
        formData.append('files', file)
      })

      const response = await fetch(`/api/sessions/${sessionId}/reference-images`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        setReferenceImages(data.images || [])
        toast({
          title: 'Reference images added',
          status: 'success',
          duration: 2000,
        })
      } else {
        throw new Error('Failed to upload images')
      }
    } catch (error) {
      toast({
        title: 'Failed to upload reference images',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
      })
    } finally {
      setIsUploading(false)
      // Reset input
      event.target.value = ''
    }
  }

  const handleDeleteImage = async (imageId: string) => {
    try {
      const response = await fetch(`/api/sessions/${sessionId}/reference-images/${imageId}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        setReferenceImages(prev => prev.filter(img => img.id !== imageId))
        toast({
          title: 'Reference image removed',
          status: 'success',
          duration: 2000,
        })
      } else {
        throw new Error('Failed to delete image')
      }
    } catch (error) {
      toast({
        title: 'Failed to remove reference image',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
      })
    }
  }

  const loadPresets = async () => {
    try {
      setIsLoadingPresets(true)
      const response = await fetch('/api/preset-references')
      if (response.ok) {
        const data = await response.json()
        setPresets(data.presets || [])
      }
    } catch (error) {
      console.error('Failed to load presets:', error)
    } finally {
      setIsLoadingPresets(false)
    }
  }

  const handleLoadPreset = async (presetName: string) => {
    try {
      setIsLoadingPreset(true)
      const response = await fetch(`/api/sessions/${sessionId}/reference-images/load-preset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ preset_name: presetName }),
      })

      if (response.ok) {
        const data = await response.json()
        setReferenceImages(data.images || [])
        toast({
          title: 'Preset loaded',
          description: data.message || `Loaded ${data.loaded_count || 0} images`,
          status: 'success',
          duration: 3000,
        })
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to load preset' }))
        throw new Error(errorData.detail || 'Failed to load preset')
      }
    } catch (error) {
      toast({
        title: 'Failed to load preset',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
      })
    } finally {
      setIsLoadingPreset(false)
    }
  }

  return (
    <Box
      w="full"
      h="full"
      bg="white"
      display="flex"
      flexDirection="column"
      borderLeft="1px"
      borderColor="gray.200"
    >
      <VStack align="stretch" spacing={4} p={6} flex={1} overflowY="auto">
        <HStack justify="space-between">
          <Heading size="md">Reference Images</Heading>
          {onClose && (
            <IconButton
              aria-label="Close panel"
              icon={<CloseIcon />}
              size="sm"
              variant="ghost"
              onClick={onClose}
            />
          )}
        </HStack>

        {/* Preset and Upload buttons */}
        <HStack spacing={2}>
          <Menu>
            <MenuButton
              as={Button}
              rightIcon={<ChevronDownIcon />}
              colorScheme="brand"
              variant="outline"
              flex={1}
              isLoading={isLoadingPresets || isLoadingPreset}
            >
              {isLoadingPreset ? 'Loading...' : 'Load Preset'}
            </MenuButton>
            <MenuList>
              {presets.length === 0 ? (
                <MenuItem isDisabled>No presets available</MenuItem>
              ) : (
                presets.map((preset) => (
                  <MenuItem
                    key={preset.name}
                    onClick={() => handleLoadPreset(preset.name)}
                  >
                    {preset.name} ({preset.image_count} images)
                  </MenuItem>
                ))
              )}
            </MenuList>
          </Menu>
          <Input
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileSelect}
            display="none"
            id="reference-image-upload"
          />
          <Button
            as="label"
            htmlFor="reference-image-upload"
            leftIcon={<AddIcon />}
            colorScheme="brand"
            variant="outline"
            flex={1}
            cursor="pointer"
            isLoading={isUploading}
          >
            Add Images
          </Button>
        </HStack>

        {/* Images grid */}
        {isLoading ? (
          <SimpleGrid columns={1} spacing={4}>
            {[1, 2, 3].map(i => (
              <Skeleton key={i} h="200px" borderRadius="md" />
            ))}
          </SimpleGrid>
        ) : referenceImages.length === 0 ? (
          <Box textAlign="center" py={8} color="gray.400">
            <Text>No reference images</Text>
            <Text fontSize="sm" mt={2}>
              Add images to use as color reference
            </Text>
          </Box>
        ) : (
          <SimpleGrid columns={1} spacing={4}>
            {referenceImages.map((img) => (
              <Box
                key={img.id}
                position="relative"
                borderRadius="md"
                overflow="hidden"
                border="1px"
                borderColor="gray.200"
                _hover={{
                  borderColor: 'brand.300',
                  boxShadow: 'md',
                }}
                transition="all 0.2s"
              >
                  <Image
                  src={img.url}
                  alt={img.name}
                  w="full"
                  maxH="400px"
                  objectFit="contain"
                  fallbackSrc="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ddd' width='100' height='100'/%3E%3C/svg%3E"
                />
                <Box
                  position="absolute"
                  top={2}
                  right={2}
                  bg="blackAlpha.700"
                  borderRadius="full"
                >
                  <IconButton
                    aria-label="Remove image"
                    icon={<CloseIcon />}
                    size="xs"
                    variant="ghost"
                    colorScheme="red"
                    onClick={() => handleDeleteImage(img.id)}
                  />
                </Box>
                <Box p={2} bg="white">
                  <Text fontSize="sm" noOfLines={1}>
                    {img.name}
                  </Text>
                </Box>
              </Box>
            ))}
          </SimpleGrid>
        )}
      </VStack>
    </Box>
  )
}

export default ReferencePanel

