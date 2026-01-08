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
import { updateMask, getImageUrl, getMaskUrl, getStage2Results, wandSelect } from '../api/client'
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
  const [isWandMode, setIsWandMode] = useState(false)
  const [isWandLoading, setIsWandLoading] = useState(false)
  const [maskHistory, setMaskHistory] = useState<ImageData[]>([])
  const lastPaintPos = useRef<{ x: number; y: number } | null>(null)
  const [brushPreviewPos, setBrushPreviewPos] = useState<{ x: number; y: number } | null>(null)
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
          if (result?.mask_path && !isWandMode) {
            // Load existing mask from .ascota metadata (only if not in wand mode)
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
            // Clear history when loading new image
            setMaskHistory([])
          }
        }
      }
    }
  }, [imageId, images, sessionId, stage2Results, isWandMode])
  
  // Initialize mask to all background when wand mode is activated
  useEffect(() => {
    if (isWandMode && maskCanvasRef.current && image) {
      const maskCanvas = maskCanvasRef.current
      const maskCtx = maskCanvas.getContext('2d')
      if (maskCtx) {
        // Save current state to history before initializing
        const currentMaskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
        setMaskHistory(prev => {
          // Only save if it's different from the last history entry
          if (prev.length === 0 || 
              JSON.stringify(currentMaskData.data) !== JSON.stringify(prev[prev.length - 1].data)) {
            return [...prev, currentMaskData].slice(-20)
          }
          return prev
        })
        
        // Initialize to all background (black)
        maskCtx.fillStyle = 'black'
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
        const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
        setMask(maskData)
        drawMaskOverlay()
      }
    }
  }, [isWandMode, image])
  
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
    
    // Draw brush preview circle if not painting and not in wand mode
    if (brushPreviewPos && !isPainting && !isWandMode) {
      // brushPreviewPos is already in canvas coordinates
      const previewX = brushPreviewPos.x
      const previewY = brushPreviewPos.y
      const previewRadius = brushSize
      
      ctx.strokeStyle = paintMode === 'in' ? '#00ff00' : '#ff0000'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.arc(previewX, previewY, previewRadius, 0, Math.PI * 2)
      ctx.stroke()
      ctx.setLineDash([])
    }
  }
  
  useEffect(() => {
    drawMaskOverlay()
  }, [maskOpacity, mask, image, brushPreviewPos, isPainting, isWandMode, brushSize, paintMode])
  
  const paintAt = (x: number, y: number, isFirstPoint: boolean = false) => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas || !mask) return
    
    const ctx = maskCanvas.getContext('2d')
    if (!ctx) return
    
    ctx.fillStyle = paintMode === 'in' ? 'white' : 'black'
    ctx.strokeStyle = paintMode === 'in' ? 'white' : 'black'
    ctx.lineWidth = brushSize * 2
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    
    if (isFirstPoint || !lastPaintPos.current) {
      // First point: draw a circle
      ctx.beginPath()
      ctx.arc(x, y, brushSize, 0, Math.PI * 2)
      ctx.fill()
      lastPaintPos.current = { x, y }
    } else {
      // Subsequent points: draw a line from previous to current
      const prev = lastPaintPos.current
      ctx.beginPath()
      ctx.moveTo(prev.x, prev.y)
      ctx.lineTo(x, y)
      ctx.stroke()
      
      // Also draw a circle at the end to ensure smooth connection
      ctx.beginPath()
      ctx.arc(x, y, brushSize, 0, Math.PI * 2)
      ctx.fill()
      
      lastPaintPos.current = { x, y }
    }
    
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
    
    if (isWandMode) {
      // Wand tool: call MobileSAM
      handleWandClick(x, y)
    } else {
      // Brush tool: paint
      setIsPainting(true)
      lastPaintPos.current = null // Reset position for new stroke
      paintAt(x, y, true)
    }
  }
  
  const handleWandClick = async (x: number, y: number) => {
    if (isWandLoading || !maskCanvasRef.current || !mask) return
    
    const maskCanvas = maskCanvasRef.current
    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return
    
    // Save current mask to history before wand operation
    const currentMaskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
    setMaskHistory(prev => {
      const newHistory = [...prev, currentMaskData]
      // Limit history to 20 states
      return newHistory.slice(-20)
    })
    
    setIsWandLoading(true)
    
    try {
      // Call MobileSAM API
      const response = await wandSelect(sessionId, imageId, Math.round(x), Math.round(y))
      
      // Load the returned mask as an image
      const samMaskImg = new window.Image()
      samMaskImg.onload = () => {
        // Create a temporary canvas to process the SAM mask
        const tempCanvas = document.createElement('canvas')
        tempCanvas.width = maskCanvas.width
        tempCanvas.height = maskCanvas.height
        const tempCtx = tempCanvas.getContext('2d')
        
        if (tempCtx) {
          // Draw the SAM mask
          tempCtx.drawImage(samMaskImg, 0, 0)
          const samMaskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
          
          // Merge SAM mask with current mask using OR operation
          // SAM mask: white = foreground (255), black = background (0)
          // Current mask: white = foreground (255), black = background (0)
          // We want to add SAM foreground to current mask
          const currentMaskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
          
          for (let i = 0; i < currentMaskData.data.length; i += 4) {
            const samValue = samMaskData.data[i] // Red channel
            const currentValue = currentMaskData.data[i]
            
            // OR operation: if either mask has foreground, result is foreground
            const mergedValue = Math.max(samValue, currentValue)
            currentMaskData.data[i] = mergedValue     // R
            currentMaskData.data[i + 1] = mergedValue // G
            currentMaskData.data[i + 2] = mergedValue // B
            currentMaskData.data[i + 3] = 255        // A
          }
          
          // Update mask canvas
          maskCtx.putImageData(currentMaskData, 0, 0)
          setMask(currentMaskData)
          drawMaskOverlay()
        }
        
        setIsWandLoading(false)
      }
      
      samMaskImg.onerror = () => {
        toast({
          title: 'Failed to load SAM mask',
          status: 'error',
          duration: 3000,
        })
        setIsWandLoading(false)
      }
      
      // Load mask from base64
      samMaskImg.src = `data:image/png;base64,${response.mask_data}`
      
    } catch (error) {
      toast({
        title: 'Wand selection failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
      })
      setIsWandLoading(false)
    }
  }
  
  const handleUndo = () => {
    if (maskHistory.length === 0 || !maskCanvasRef.current) return
    
    const maskCanvas = maskCanvasRef.current
    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return
    
    // Restore previous mask state
    const previousMask = maskHistory[maskHistory.length - 1]
    maskCtx.putImageData(previousMask, 0, 0)
    setMask(previousMask)
    setMaskHistory(prev => prev.slice(0, -1))
    drawMaskOverlay()
  }
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const displayX = e.clientX - rect.left
    const displayY = e.clientY - rect.top
    const canvasX = displayX * scaleX
    const canvasY = displayY * scaleY
    
    // Update brush preview position (in canvas coordinates)
    if (!isPainting && !isWandMode) {
      setBrushPreviewPos({ x: canvasX, y: canvasY })
    } else {
      setBrushPreviewPos(null)
    }
    
    // Handle painting
    if (isPainting) {
      paintAt(canvasX, canvasY)
    }
  }
  
  const handleMouseUp = () => {
    setIsPainting(false)
    lastPaintPos.current = null // Reset position when stroke ends
  }
  
  const handleMouseLeave = () => {
    setIsPainting(false)
    lastPaintPos.current = null
    setBrushPreviewPos(null) // Hide preview when mouse leaves canvas
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
      onSuccess: async () => {
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
          onClick={() => {
            setIsWandMode(false)
            setPaintMode('in')
          }}
          colorScheme={!isWandMode && paintMode === 'in' ? 'green' : 'gray'}
          isDisabled={isWandMode}
        >
          Paint In (Foreground)
        </Button>
        <Button
          onClick={() => {
            setIsWandMode(false)
            setPaintMode('out')
          }}
          colorScheme={!isWandMode && paintMode === 'out' ? 'red' : 'gray'}
          isDisabled={isWandMode}
        >
          Paint Out (Background)
        </Button>
        <Button
          onClick={() => setIsWandMode(true)}
          colorScheme={isWandMode ? 'purple' : 'gray'}
          isLoading={isWandLoading}
        >
          Wand Tool
        </Button>
        <Button
          onClick={handleUndo}
          isDisabled={maskHistory.length === 0 || isWandLoading}
          variant="outline"
        >
          Undo
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
      
      <Box 
        border="1px" 
        borderColor="gray.200" 
        borderRadius="md" 
        overflow="auto"
        maxH="calc(100vh - 350px)"
        display="flex"
        justifyContent="center"
        alignItems="flex-start"
        bg="gray.50"
        p={2}
      >
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          style={{ 
            cursor: isWandMode ? (isWandLoading ? 'wait' : 'pointer') : 'crosshair', 
            display: 'block', 
            maxWidth: '100%', 
            maxHeight: 'calc(100vh - 350px)',
            width: 'auto',
            height: 'auto'
          }}
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

