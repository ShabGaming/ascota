import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Button,
  HStack,
  VStack,
  useToast,
  Text,
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

/**
 * Calculate perspective transform matrix from 4 source points to 4 destination points.
 * Returns a 3x3 matrix as a flat array [a, b, c, d, e, f, g, h, i]
 * Based on OpenCV's getPerspectiveTransform implementation.
 */
function getPerspectiveTransform(
  srcPoints: number[][],
  dstPoints: number[][]
): number[] {
  // Build system of equations: Ax = b
  const A: number[][] = []
  const b: number[] = []
  
  for (let i = 0; i < 4; i++) {
    const [x, y] = srcPoints[i]
    const [u, v] = dstPoints[i]
    
    // For x coordinate transformation: u = (a*x + b*y + c) / (g*x + h*y + 1)
    // Rearranged: a*x + b*y + c - u*g*x - u*h*y = u
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y])
    b.push(u)
    
    // For y coordinate transformation: v = (d*x + e*y + f) / (g*x + h*y + 1)
    // Rearranged: d*x + e*y + f - v*g*x - v*h*y = v
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y])
    b.push(v)
  }
  
  // Solve 8x8 system using Gaussian elimination with partial pivoting
  const n = 8
  const augmented: number[][] = A.map((row, i) => [...row, b[i]])
  
  // Forward elimination with partial pivoting
  for (let i = 0; i < n; i++) {
    // Find pivot row
    let maxRow = i
    let maxVal = Math.abs(augmented[i][i])
    for (let k = i + 1; k < n; k++) {
      const val = Math.abs(augmented[k][i])
      if (val > maxVal) {
        maxVal = val
        maxRow = k
      }
    }
    
    // Swap rows if needed
    if (maxRow !== i) {
      ;[augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]]
    }
    
    // Skip if pivot is zero (shouldn't happen for valid perspective transform)
    if (Math.abs(augmented[i][i]) < 1e-10) {
      throw new Error('Singular matrix in perspective transform calculation')
    }
    
    // Eliminate column below pivot
    for (let k = i + 1; k < n; k++) {
      const factor = augmented[k][i] / augmented[i][i]
      for (let j = i; j < n + 1; j++) {
        augmented[k][j] -= factor * augmented[i][j]
      }
    }
  }
  
  // Back substitution
  const x = new Array(n)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i][n]
    for (let j = i + 1; j < n; j++) {
      x[i] -= augmented[i][j] * x[j]
    }
    x[i] /= augmented[i][i]
  }
  
  // Return 3x3 matrix [a, b, c, d, e, f, g, h, 1]
  // Note: i (last element) is always 1 for perspective transform
  return [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1]
}

/**
 * Apply perspective transform to a point.
 * @param point [x, y] in source coordinates
 * @param matrix 3x3 transform matrix as flat array
 * @returns [x, y] in destination coordinates
 */
function perspectiveTransform(
  point: number[],
  matrix: number[]
): number[] {
  const [x, y] = point
  const [a, b, c, d, e, f, g, h, i] = matrix
  
  const denominator = g * x + h * y + i
  const u = (a * x + b * y + c) / denominator
  const v = (d * x + e * y + f) / denominator
  
  return [u, v]
}

function ScaleEditor({ sessionId, imageId, onClose }: ScaleEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const magnifyCanvasRef = useRef<HTMLCanvasElement>(null)
  const [cardImage, setCardImage] = useState<HTMLImageElement | null>(null)
  const [centers, setCenters] = useState<Point[]>([])
  const [scale, setScale] = useState(1)
  const [baseCanvasWidth, setBaseCanvasWidth] = useState(0)
  const [baseCanvasHeight, setBaseCanvasHeight] = useState(0)
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null)
  const [magnifyZoom] = useState(3) // Fixed zoom level for magnified view
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
        // Maintain aspect ratio, fit to larger size for better editing
        const maxWidth = 1200
        const maxHeight = 800
        const imgAspect = img.width / img.height
        const maxAspect = maxWidth / maxHeight
        
        let canvasWidth = maxWidth
        let canvasHeight = maxHeight
        if (imgAspect > maxAspect) {
          canvasHeight = maxWidth / imgAspect
        } else {
          canvasWidth = maxHeight * imgAspect
        }
        
        // Store base dimensions for zoom calculations
        setBaseCanvasWidth(canvasWidth)
        setBaseCanvasHeight(canvasHeight)
        
        canvas.width = canvasWidth
        canvas.height = canvasHeight
        setScale(canvasWidth / img.width)
        
        // Transform centers from original image coordinates to card crop coordinates
        if (stage3Data.centers && stage3Data.centers.length === 3) {
          // Centers are in original image coordinates
          // We need to transform them to card crop coordinates using perspective transform
          const cardCoords = hybridCard.coordinates
          if (cardCoords && cardCoords.length === 4) {
            // Calculate card crop dimensions (same as backend)
            const xCoords = cardCoords.map(c => c[0])
            const yCoords = cardCoords.map(c => c[1])
            const cropWidth = Math.max(...xCoords) - Math.min(...xCoords)
            const cropHeight = Math.max(...yCoords) - Math.min(...yCoords)
            
            // Destination points for perspective transform (rectangular crop)
            // This matches what the backend uses: [0,0], [w,0], [w,h], [0,h]
            const dstPoints = [
              [0, 0],
              [cropWidth, 0],
              [cropWidth, cropHeight],
              [0, cropHeight]
            ]
            
            // Calculate perspective transform matrix from original image to card crop
            const transformMatrix = getPerspectiveTransform(cardCoords, dstPoints)
            
            // Transform centers from original image coordinates to card crop coordinates
            const transformedCenters = stage3Data.centers.map((center: number[]) => {
              const [cropX, cropY] = perspectiveTransform(center, transformMatrix)
              
              // Scale to canvas coordinates (card crop image dimensions should match cropWidth/cropHeight)
              return {
                x: (cropX / cropWidth) * canvasWidth,
                y: (cropY / cropHeight) * canvasHeight
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
      ctx.drawImage(cardImage, 0, 0, baseCanvasWidth, baseCanvasHeight)
    } else {
      // Draw placeholder
      ctx.fillStyle = 'lightgray'
      ctx.fillRect(0, 0, baseCanvasWidth, baseCanvasHeight)
      ctx.fillStyle = 'black'
      ctx.font = '16px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('Card crop will appear here', baseCanvasWidth / 2, baseCanvasHeight / 2)
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
  
  const drawMagnify = () => {
    const canvas = magnifyCanvasRef.current
    if (!canvas || !cardImage || draggingIndex === null) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const center = centers[draggingIndex]
    const magnifySize = 200 // Size of the magnified view
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw background
    ctx.fillStyle = '#f0f0f0'
    ctx.fillRect(0, 0, magnifySize, magnifySize)
    
    // Draw border
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 2
    ctx.strokeRect(0, 0, magnifySize, magnifySize)
    
    // Save context
    ctx.save()
    
    // Clip to magnify area
    ctx.beginPath()
    ctx.rect(0, 0, magnifySize, magnifySize)
    ctx.clip()
    
    // Translate and scale to show the area around the center point
    ctx.translate(magnifySize / 2, magnifySize / 2)
    ctx.scale(magnifyZoom, magnifyZoom)
    ctx.translate(-center.x, -center.y)
    
    // Draw the card image
    ctx.drawImage(cardImage, 0, 0, baseCanvasWidth, baseCanvasHeight)
    
    // Draw all centers
    centers.forEach((c, idx) => {
      ctx.fillStyle = idx === draggingIndex ? 'red' : 'lime'
      ctx.beginPath()
      ctx.arc(c.x, c.y, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.stroke()
      
      // Draw crosshairs
      ctx.strokeStyle = 'red'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(c.x - 15, c.y)
      ctx.lineTo(c.x + 15, c.y)
      ctx.moveTo(c.x, c.y - 15)
      ctx.lineTo(c.x, c.y + 15)
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
    
    ctx.restore()
    
    // Draw center indicator
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(magnifySize / 2 - 10, magnifySize / 2)
    ctx.lineTo(magnifySize / 2 + 10, magnifySize / 2)
    ctx.moveTo(magnifySize / 2, magnifySize / 2 - 10)
    ctx.lineTo(magnifySize / 2, magnifySize / 2 + 10)
    ctx.stroke()
    ctx.setLineDash([])
  }
  
  useEffect(() => {
    draw()
  }, [centers, cardImage, draggingIndex, baseCanvasWidth, baseCanvasHeight])
  
  useEffect(() => {
    drawMagnify()
  }, [centers, cardImage, draggingIndex, baseCanvasWidth, baseCanvasHeight, magnifyZoom])
  
  const getPointAt = (x: number, y: number): number | null => {
    // x and y are already in base canvas coordinates
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
    if (!canvas || baseCanvasWidth === 0 || baseCanvasHeight === 0) return
    
    const rect = canvas.getBoundingClientRect()
    // Convert screen coordinates to base canvas coordinates
    // Account for CSS scaling (canvas may be scaled to fit container)
    const scaleX = baseCanvasWidth / rect.width
    const scaleY = baseCanvasHeight / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY
    
    const index = getPointAt(x, y)
    if (index !== null) {
      setDraggingIndex(index)
    }
  }
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingIndex === null) return
    
    const canvas = canvasRef.current
    if (!canvas || baseCanvasWidth === 0 || baseCanvasHeight === 0) return
    
    const rect = canvas.getBoundingClientRect()
    // Convert screen coordinates to base canvas coordinates
    // Account for CSS scaling (canvas may be scaled to fit container)
    const scaleX = baseCanvasWidth / rect.width
    const scaleY = baseCanvasHeight / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY
    
    const newCenters = [...centers]
    newCenters[draggingIndex] = { x, y }
    setCenters(newCenters)
  }
  
  const handleMouseUp = () => {
    setDraggingIndex(null)
  }
  
  const updateMutation = useMutation(
    () => {
      // Centers are in canvas coordinates (already accounting for zoom in the drawing)
      // Need to convert to card crop coordinates
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
        Drag the three circle centers to adjust their positions. Hold a point to see a magnified view on the right.
      </Text>
      
      <Box position="relative" width="100%">
        <Box 
          border="1px" 
          borderColor="gray.200" 
          borderRadius="md" 
          overflow="auto"
          display="flex" 
          justifyContent="center" 
          alignItems="center"
          maxH="70vh"
        >
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{ 
              cursor: 'crosshair', 
              display: 'block',
              maxWidth: '100%',
              height: 'auto',
              width: 'auto',
            }}
          />
        </Box>
        
        {draggingIndex !== null && (
          <Box
            position="absolute"
            top={0}
            right={4}
            zIndex={10}
            bg="white"
            border="1px"
            borderColor="gray.300"
            borderRadius="md"
            p={3}
            boxShadow="lg"
            minW="220px"
          >
            <VStack spacing={2} align="stretch">
              <Text fontSize="sm" fontWeight="bold" color="gray.700">
                Magnified View ({magnifyZoom}x)
              </Text>
              <Box 
                border="1px" 
                borderColor="gray.200" 
                borderRadius="md"
                bg="white"
                p={2}
              >
                <canvas
                  ref={magnifyCanvasRef}
                  width={200}
                  height={200}
                  style={{ 
                    display: 'block',
                    width: '200px',
                    height: '200px',
                  }}
                />
              </Box>
              <Text fontSize="xs" color="gray.500">
                Hold and drag to adjust position
              </Text>
            </VStack>
          </Box>
        )}
      </Box>
      
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

