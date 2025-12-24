import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Button,
  HStack,
  VStack,
  useToast,
  Text,
  Stat,
  StatLabel,
  StatNumber,
} from '@chakra-ui/react'
import { useMutation, useQueryClient } from 'react-query'
import { updateCenters, getCardCropUrl, getStage3Results } from '../api/client'
import { useSessionStore } from '../state/session'

interface ScaleEditorProps {
  sessionId: string
  imageId: string
  onClose: () => void
}

interface Point {
  x: number
  y: number
}

function ScaleEditor({ sessionId, imageId, onClose }: ScaleEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [cardImage, setCardImage] = useState<HTMLImageElement | null>(null)
  const [centers, setCenters] = useState<Point[]>([])
  const [scale, setScale] = useState(1)
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null)
  const [calculatedPxPerCm, setCalculatedPxPerCm] = useState<number | null>(null)
  const toast = useToast()
  const queryClient = useQueryClient()
  
  const stage1Results = useSessionStore(state => state.stage1Results)
  const stage3Results = useSessionStore(state => state.stage3Results)
  const setStage3Results = useSessionStore(state => state.setStage3Results)
  
  // Load card crop and centers
  useEffect(() => {
    const stage1Data = stage1Results[imageId]
    const stage3Data = stage3Results[imageId]
    
    if (!stage1Data || !stage3Data) return
    
    // Find 8-hybrid card
    const hybridCard = stage1Data.cards?.find(card => card.card_type === '8_hybrid_card')
    if (!hybridCard) return
    
    // Load card crop image
    const img = new window.Image()
    img.crossOrigin = 'anonymous'
    img.src = getCardCropUrl(sessionId, imageId)
    
    img.onload = () => {
      setCardImage(img)
      
      // Set canvas size to match image
      const canvas = canvasRef.current
      if (canvas) {
        // Maintain aspect ratio, fit to max 600x400
        const maxWidth = 600
        const maxHeight = 400
        const imgAspect = img.width / img.height
        const maxAspect = maxWidth / maxHeight
        
        let canvasWidth = maxWidth
        let canvasHeight = maxHeight
        if (imgAspect > maxAspect) {
          canvasHeight = maxWidth / imgAspect
        } else {
          canvasWidth = maxHeight * imgAspect
        }
        
        canvas.width = canvasWidth
        canvas.height = canvasHeight
        setScale(canvasWidth / img.width)
        
        // Transform centers from original image coordinates to card crop coordinates
        if (stage3Data.centers && stage3Data.centers.length === 3) {
          // Centers are in original image coordinates
          // We need to transform them to card crop coordinates
          // For now, we'll use a simple approach: if centers are within the card bounds, scale them
          // This is a simplified transformation - in production, you'd use proper perspective transform
          const cardCoords = hybridCard.coordinates
          if (cardCoords && cardCoords.length === 4) {
            // Get card bounding box
            const xCoords = cardCoords.map(c => c[0])
            const yCoords = cardCoords.map(c => c[1])
            const minX = Math.min(...xCoords)
            const maxX = Math.max(...xCoords)
            const minY = Math.min(...yCoords)
            const maxY = Math.max(...yCoords)
            const cardWidth = maxX - minX
            const cardHeight = maxY - minY
            
            // Transform centers to card crop coordinates (assuming rectangular crop)
            const transformedCenters = stage3Data.centers.map((center: number[]) => {
              // Relative position within card bounds
              const relX = (center[0] - minX) / cardWidth
              const relY = (center[1] - minY) / cardHeight
              
              // Scale to canvas coordinates
              return {
                x: relX * canvasWidth,
                y: relY * canvasHeight
              }
            })
            
            setCenters(transformedCenters)
          }
        } else {
          // No centers available, use default positions
          setCenters([
            { x: canvasWidth / 2 - 50, y: canvasHeight / 2 },
            { x: canvasWidth / 2 + 50, y: canvasHeight / 2 },
            { x: canvasWidth / 2, y: canvasHeight / 2 - 30 },
          ])
        }
      }
    }
    
    img.onerror = () => {
      console.error('Failed to load card crop image')
    }
  }, [imageId, sessionId, stage1Results, stage3Results])
  
  const draw = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw card image if available
    if (cardImage) {
      ctx.drawImage(cardImage, 0, 0, canvas.width, canvas.height)
    } else {
      // Draw placeholder
      ctx.fillStyle = 'lightgray'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = 'black'
      ctx.font = '16px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('Card crop will appear here', canvas.width / 2, canvas.height / 2)
    }
    
    // Draw centers
    centers.forEach((center, idx) => {
      ctx.fillStyle = idx === draggingIndex ? 'red' : 'lime'
      ctx.beginPath()
      ctx.arc(center.x, center.y, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.stroke()
      
      // Draw crosshairs
      ctx.strokeStyle = 'red'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(center.x - 15, center.y)
      ctx.lineTo(center.x + 15, center.y)
      ctx.moveTo(center.x, center.y - 15)
      ctx.lineTo(center.x, center.y + 15)
      ctx.stroke()
    })
    
    // Draw connecting lines
    if (centers.length === 3) {
      ctx.strokeStyle = 'cyan'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(centers[0].x, centers[0].y)
      ctx.lineTo(centers[1].x, centers[1].y)
      ctx.lineTo(centers[2].x, centers[2].y)
      ctx.closePath()
      ctx.stroke()
    }
  }
  
  useEffect(() => {
    draw()
  }, [centers, cardImage, draggingIndex])
  
  const getPointAt = (x: number, y: number): number | null => {
    for (let i = 0; i < centers.length; i++) {
      const center = centers[i]
      const dist = Math.sqrt((x - center.x) ** 2 + (y - center.y) ** 2)
      if (dist < 15) {
        return i
      }
    }
    return null
  }
  
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    const index = getPointAt(x, y)
    if (index !== null) {
      setDraggingIndex(index)
    }
  }
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingIndex === null) return
    
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    const newCenters = [...centers]
    newCenters[draggingIndex] = { x, y }
    setCenters(newCenters)
  }
  
  const handleMouseUp = () => {
    setDraggingIndex(null)
  }
  
  const updateMutation = useMutation(
    () => {
      // Centers are in canvas coordinates, need to convert to card crop coordinates
      // Canvas is scaled to fit the card crop, so divide by scale to get crop coordinates
      const centersArray = centers.map(c => [c.x / scale, c.y / scale])
      return updateCenters(sessionId, imageId, centersArray)
    },
    {
      onSuccess: async (data) => {
        // Invalidate and refetch stage3 results to update preview
        await queryClient.invalidateQueries(['stage3Results', sessionId])
        // Force a refetch to ensure the preview updates
        const updatedResults = await getStage3Results(sessionId)
        setStage3Results(updatedResults.results)
        // Also trigger query refetch for any components using the query
        await queryClient.refetchQueries(['stage3Results', sessionId])
        
        toast({
          title: 'Centers updated',
          description: `Scale: ${data.pixels_per_cm.toFixed(0)} px/cm`,
          status: 'success',
          duration: 2000,
        })
        
        // Close the editor
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
      <Text fontSize="sm" color="gray.600">
        Drag the three circle centers to adjust their positions. Click "Recalculate" to update the scale.
      </Text>
      
      <Box border="1px" borderColor="gray.200" borderRadius="md" overflow="hidden">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ cursor: 'crosshair', display: 'block' }}
          width={600}
          height={400}
        />
      </Box>
      
      {calculatedPxPerCm && (
        <Stat>
          <StatLabel>Calculated Pixels per cm</StatLabel>
          <StatNumber>{calculatedPxPerCm.toFixed(0)}</StatNumber>
        </Stat>
      )}
      
      <HStack justify="flex-end">
        <Button onClick={onClose}>Cancel</Button>
        <Button
          colorScheme="brand"
          onClick={() => updateMutation.mutate()}
          isLoading={updateMutation.isLoading}
        >
          Recalculate & Save
        </Button>
      </HStack>
    </VStack>
  )
}

export default ScaleEditor

