import { useEffect, useRef, useState } from 'react'
import { Box } from '@chakra-ui/react'
import { getImageUrl, ImageItem, ScaleResult, CardDetection } from '../api/client'

interface ImageWithScaleCardProps {
  sessionId: string
  image: ImageItem
  result?: ScaleResult
  card?: CardDetection
  maxHeight?: string
}

function ImageWithScaleCard({ sessionId, image, result, card, maxHeight = '200px' }: ImageWithScaleCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Load image
    const img = new window.Image()
    img.crossOrigin = 'anonymous'
    img.src = getImageUrl(sessionId, image.id, '3000')
    
    img.onload = () => {
      // Calculate display size (maintain aspect ratio)
      const maxDisplayHeight = parseInt(maxHeight)
      const scale = Math.min(1, maxDisplayHeight / img.height)
      const displayWidth = img.width * scale
      const displayHeight = img.height * scale
      
      canvas.width = displayWidth
      canvas.height = displayHeight
      
      // Draw image
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight)
      
      setImageLoaded(true)
      
      // Draw card visualization based on method
      if (result && card && result.method) {
        const imageScale = displayWidth / img.width
        
        if (result.method === 'checker_card') {
          // Draw bounding box around checker card
          if (card.coordinates && card.coordinates.length === 4) {
            ctx.strokeStyle = '#00ff00' // Green
            ctx.lineWidth = 3
            ctx.beginPath()
            
            // Scale coordinates to display size
            const scaledCoords = card.coordinates.map(([x, y]) => [
              x * imageScale,
              y * imageScale
            ])
            
            ctx.moveTo(scaledCoords[0][0], scaledCoords[0][1])
            for (let i = 1; i < scaledCoords.length; i++) {
              ctx.lineTo(scaledCoords[i][0], scaledCoords[i][1])
            }
            ctx.closePath()
            ctx.stroke()
            
            // Add semi-transparent fill
            ctx.fillStyle = 'rgba(0, 255, 0, 0.2)'
            ctx.fill()
          }
        } else if (result.method === '8_hybrid_card') {
          // Draw card bounding box
          if (card.coordinates && card.coordinates.length === 4) {
            ctx.strokeStyle = '#00ffff' // Cyan
            ctx.lineWidth = 2
            ctx.beginPath()
            
            // Scale coordinates to display size
            const scaledCoords = card.coordinates.map(([x, y]) => [
              x * imageScale,
              y * imageScale
            ])
            
            ctx.moveTo(scaledCoords[0][0], scaledCoords[0][1])
            for (let i = 1; i < scaledCoords.length; i++) {
              ctx.lineTo(scaledCoords[i][0], scaledCoords[i][1])
            }
            ctx.closePath()
            ctx.stroke()
            
            // Add semi-transparent fill
            ctx.fillStyle = 'rgba(0, 255, 255, 0.15)'
            ctx.fill()
          }
          
          // Draw circle centers if available
          if (result.centers && result.centers.length === 3) {
            result.centers.forEach((center: number[], idx: number) => {
              const x = center[0] * imageScale
              const y = center[1] * imageScale
              
              // Draw circle
              ctx.fillStyle = '#ffff00' // Yellow
              ctx.beginPath()
              ctx.arc(x, y, 8, 0, Math.PI * 2)
              ctx.fill()
              
              // Draw white border
              ctx.strokeStyle = '#ffffff'
              ctx.lineWidth = 2
              ctx.stroke()
              
              // Draw crosshairs
              ctx.strokeStyle = '#ff0000' // Red
              ctx.lineWidth = 2
              ctx.beginPath()
              ctx.moveTo(x - 15, y)
              ctx.lineTo(x + 15, y)
              ctx.moveTo(x, y - 15)
              ctx.lineTo(x, y + 15)
              ctx.stroke()
            })
            
            // Draw connecting lines between centers
            if (result.centers.length === 3) {
              ctx.strokeStyle = '#00ff00' // Green
              ctx.lineWidth = 2
              ctx.beginPath()
              const centers = result.centers.map((c: number[]) => [
                c[0] * imageScale,
                c[1] * imageScale
              ])
              ctx.moveTo(centers[0][0], centers[0][1])
              ctx.lineTo(centers[1][0], centers[1][1])
              ctx.lineTo(centers[2][0], centers[2][1])
              ctx.closePath()
              ctx.stroke()
            }
          }
        }
      }
    }
    
    img.onerror = () => {
      // Draw error placeholder
      ctx.fillStyle = '#f0f0f0'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#666'
      ctx.font = '14px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('Failed to load image', canvas.width / 2, canvas.height / 2)
    }
  }, [sessionId, image.id, result, card, maxHeight])
  
  return (
    <Box position="relative" width="100%" bg="gray.100" display="flex" justifyContent="center" alignItems="center">
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
          maxWidth: '100%',
          maxHeight: maxHeight,
          height: 'auto',
          width: 'auto',
        }}
      />
    </Box>
  )
}

export default ImageWithScaleCard

