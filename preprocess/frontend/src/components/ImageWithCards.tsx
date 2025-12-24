import { useEffect, useRef, useState } from 'react'
import { Box } from '@chakra-ui/react'
import { getImageUrl, ImageItem, ImageCardResult } from '../api/client'

interface ImageWithCardsProps {
  sessionId: string
  image: ImageItem
  result?: ImageCardResult
  maxHeight?: string
}

function ImageWithCards({ sessionId, image, result, maxHeight = '200px' }: ImageWithCardsProps) {
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
      
      // Draw card overlays if available
      if (result?.cards && result.cards.length > 0) {
        const imageScale = scale // Scale from original to display
        
        result.cards.forEach((card) => {
          const color = getCardColor(card.card_type)
          const rgb = getColorRGB(color)
          
          // Draw polygon
          ctx.strokeStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`
          ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.2)`
          ctx.lineWidth = 2
          
          ctx.beginPath()
          if (card.coordinates && card.coordinates.length >= 4) {
            // Scale coordinates to display size
            const firstPoint = card.coordinates[0]
            ctx.moveTo(firstPoint[0] * imageScale, firstPoint[1] * imageScale)
            
            for (let i = 1; i < card.coordinates.length; i++) {
              const point = card.coordinates[i]
              ctx.lineTo(point[0] * imageScale, point[1] * imageScale)
            }
            ctx.closePath()
            ctx.fill()
            ctx.stroke()
          }
        })
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
  }, [sessionId, image.id, result, maxHeight])
  
  const getCardColor = (cardType: string) => {
    switch (cardType) {
      case '24_color_card':
        return 'green'
      case '8_hybrid_card':
        return 'blue'
      case 'checker_card':
        return 'red'
      default:
        return 'gray'
    }
  }
  
  const getColorRGB = (color: string) => {
    switch (color) {
      case 'green':
        return { r: 0, g: 255, b: 0 }
      case 'blue':
        return { r: 0, g: 0, b: 255 }
      case 'red':
        return { r: 255, g: 0, b: 0 }
      default:
        return { r: 128, g: 128, b: 128 }
    }
  }
  
  return (
    <Box position="relative" width="100%" bg="gray.100">
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
          width: '100%',
          height: 'auto',
          maxHeight: maxHeight,
        }}
      />
    </Box>
  )
}

export default ImageWithCards

