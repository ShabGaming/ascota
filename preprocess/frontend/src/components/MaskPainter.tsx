import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Button,
  HStack,
  VStack,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  FormLabel,
  useToast,
} from '@chakra-ui/react'
import { useMutation, useQueryClient } from 'react-query'
import { updateMask, getImageUrl, getMaskUrl, getStage2Results } from '../api/client'
import { useSessionStore } from '../state/session'

interface MaskPainterProps {
  sessionId: string
  imageId: string
  onClose: () => void
}

function MaskPainter({ sessionId, imageId, onClose }: MaskPainterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [mask, setMask] = useState<ImageData | null>(null)
  const [brushSize, setBrushSize] = useState(20)
  const [isPainting, setIsPainting] = useState(false)
  const [paintMode, setPaintMode] = useState<'in' | 'out'>('in')
  const [maskOpacity, setMaskOpacity] = useState(0.5)
  const toast = useToast()
  
  const images = useSessionStore(state => state.images)
  const stage2Results = useSessionStore(state => state.stage2Results)
  const setStage2Results = useSessionStore(state => state.setStage2Results)
  const queryClient = useQueryClient()
  
  // Load image and mask
  useEffect(() => {
    const imageItem = images[imageId]
    if (!imageItem) return
    
    const img = new window.Image()
    img.crossOrigin = 'anonymous'
    // Use API endpoint to serve image
    img.src = getImageUrl(sessionId, imageId, '3000')
    
    img.onload = () => {
      setImage(img)
      const canvas = canvasRef.current
      const maskCanvas = maskCanvasRef.current
      if (canvas && maskCanvas) {
        canvas.width = img.width
        canvas.height = img.height
        maskCanvas.width = img.width
        maskCanvas.height = img.height
        
        // Draw image
        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.drawImage(img, 0, 0)
        }
        
        // Initialize mask (white = foreground)
        const maskCtx = maskCanvas.getContext('2d')
        if (maskCtx) {
          const result = stage2Results[imageId]
          if (result?.mask_path) {
            // Load existing mask from .ascota metadata
            const maskImg = new window.Image()
            maskImg.crossOrigin = 'anonymous'
            maskImg.src = getMaskUrl(sessionId, imageId, result.mask_path)
            maskImg.onload = () => {
              maskCtx.drawImage(maskImg, 0, 0)
              const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
              setMask(maskData)
              // Draw overlay after mask is loaded
              setTimeout(() => drawMaskOverlay(), 0)
            }
            maskImg.onerror = () => {
              // If mask fails to load, initialize empty mask
              maskCtx.fillStyle = 'black'
              maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
              const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
              setMask(maskData)
            }
          } else {
            // Initialize empty mask (all background)
            maskCtx.fillStyle = 'black'
            maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
            const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
            setMask(maskData)
          }
        }
      }
    }
  }, [imageId, images, sessionId, stage2Results])
  
  const drawMaskOverlay = () => {
    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    if (!canvas || !maskCanvas || !mask) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Redraw image first
    if (image) {
      ctx.drawImage(image, 0, 0)
    }
    
    // Draw mask overlay with dark red tint for background
    const maskCtx = maskCanvas.getContext('2d')
    if (maskCtx) {
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
      
      // Create overlay layer on a separate canvas
      const overlayCanvas = document.createElement('canvas')
      overlayCanvas.width = maskData.width
      overlayCanvas.height = maskData.height
      const overlayCtx = overlayCanvas.getContext('2d')
      
      if (overlayCtx) {
        const overlay = overlayCtx.createImageData(maskData.width, maskData.height)
        
        for (let i = 0; i < maskData.data.length; i += 4) {
          const maskValue = maskData.data[i] / 255 // Use red channel as mask (0-1)
          
          // Background areas (mask = 0 or dark) should be tinted dark red
          // Foreground areas (mask = 1 or white) should remain unchanged
          if (maskValue < 0.5) {
            // Background: apply dark red tint overlay
            overlay.data[i] = 150     // R - dark red
            overlay.data[i + 1] = 0   // G
            overlay.data[i + 2] = 0   // B
            overlay.data[i + 3] = 255 * maskOpacity // A - use opacity slider
          } else {
            // Foreground: transparent (no overlay)
            overlay.data[i] = 0
            overlay.data[i + 1] = 0
            overlay.data[i + 2] = 0
            overlay.data[i + 3] = 0
          }
        }
        
        // Put overlay data on overlay canvas
        overlayCtx.putImageData(overlay, 0, 0)
        
        // Composite the overlay on top of the image using source-over blend mode
        ctx.globalCompositeOperation = 'source-over'
        ctx.drawImage(overlayCanvas, 0, 0)
      }
    }
  }
  
  useEffect(() => {
    drawMaskOverlay()
  }, [maskOpacity, mask, image])
  
  const paintAt = (x: number, y: number) => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas || !mask) return
    
    const ctx = maskCanvas.getContext('2d')
    if (!ctx) return
    
    ctx.fillStyle = paintMode === 'in' ? 'white' : 'black'
    ctx.beginPath()
    ctx.arc(x, y, brushSize, 0, Math.PI * 2)
    ctx.fill()
    
    // Update mask data
    const maskData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
    setMask(maskData)
    drawMaskOverlay()
  }
  
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY
    
    setIsPainting(true)
    paintAt(x, y)
  }
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isPainting) return
    
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY
    
    paintAt(x, y)
  }
  
  const handleMouseUp = () => {
    setIsPainting(false)
  }
  
  const updateMutation = useMutation(
    () => {
      const maskCanvas = maskCanvasRef.current
      if (!maskCanvas) throw new Error('No mask canvas')
      
      // Convert to base64
      const dataUrl = maskCanvas.toDataURL('image/png')
      const base64 = dataUrl.split(',')[1]
      
      return updateMask(sessionId, imageId, base64)
    },
    {
      onSuccess: async (response) => {
        // Invalidate and refetch stage2 results to update preview
        await queryClient.invalidateQueries(['stage2Results', sessionId])
        // Force a refetch to ensure the preview updates
        const updatedResults = await getStage2Results(sessionId)
        setStage2Results(updatedResults.results)
        // Also trigger query refetch for any components using the query
        await queryClient.refetchQueries(['stage2Results', sessionId])
        toast({
          title: 'Mask updated',
          status: 'success',
          duration: 2000,
        })
        onClose()
      },
      onError: (error) => {
        toast({
          title: 'Update failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          status: 'error',
          duration: 5000,
        })
      },
    }
  )
  
  return (
    <VStack spacing={4} align="stretch">
      <HStack>
        <FormLabel>Brush Size: {brushSize}px</FormLabel>
        <Slider
          value={brushSize}
          onChange={(val) => setBrushSize(val)}
          min={5}
          max={100}
          flex={1}
        >
          <SliderTrack>
            <SliderFilledTrack />
          </SliderTrack>
          <SliderThumb />
        </Slider>
      </HStack>
      
      <HStack>
        <Button
          onClick={() => setPaintMode('in')}
          colorScheme={paintMode === 'in' ? 'green' : 'gray'}
        >
          Paint In (Foreground)
        </Button>
        <Button
          onClick={() => setPaintMode('out')}
          colorScheme={paintMode === 'out' ? 'red' : 'gray'}
        >
          Paint Out (Background)
        </Button>
      </HStack>
      
      <HStack>
        <FormLabel>Mask Opacity: {(maskOpacity * 100).toFixed(0)}%</FormLabel>
        <Slider
          value={maskOpacity}
          onChange={(val) => setMaskOpacity(val)}
          min={0}
          max={1}
          step={0.01}
          flex={1}
        >
          <SliderTrack>
            <SliderFilledTrack />
          </SliderTrack>
          <SliderThumb />
        </Slider>
      </HStack>
      
      <Box border="1px" borderColor="gray.200" borderRadius="md" overflow="hidden">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ cursor: 'crosshair', display: 'block', maxWidth: '100%', height: 'auto' }}
        />
        <canvas ref={maskCanvasRef} style={{ display: 'none' }} />
      </Box>
      
      <HStack justify="flex-end">
        <Button onClick={onClose}>Cancel</Button>
        <Button
          colorScheme="brand"
          onClick={() => updateMutation.mutate()}
          isLoading={updateMutation.isLoading}
        >
          Save
        </Button>
      </HStack>
    </VStack>
  )
}

export default MaskPainter

