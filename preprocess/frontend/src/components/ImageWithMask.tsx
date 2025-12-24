import { useEffect, useRef, useState } from 'react'
import { Box } from '@chakra-ui/react'
import { getImageUrl, getMaskUrl, ImageItem, MaskResult } from '../api/client'

interface ImageWithMaskProps {
  sessionId: string
  image: ImageItem
  result?: MaskResult
  maxHeight?: string
  opacity?: number
}

function ImageWithMask({ sessionId, image, result, maxHeight = '200px', opacity = 0.5 }: ImageWithMaskProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [maskLoaded, setMaskLoaded] = useState(false)
  
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
      
      // Load and draw mask if available
      if (result?.mask_path) {
        const maskImg = new window.Image()
        maskImg.crossOrigin = 'anonymous'
        // Add cache-busting timestamp based on current time to force reload when mask is updated
        // This ensures the browser doesn't serve a cached version
        maskImg.src = getMaskUrl(sessionId, image.id, result.mask_path, Date.now())
        
        maskImg.onload = () => {
          // Redraw the image first (in case it was cleared)
          ctx.drawImage(img, 0, 0, displayWidth, displayHeight)
          
          // Create a temporary canvas to process the mask
          const tempCanvas = document.createElement('canvas')
          tempCanvas.width = displayWidth
          tempCanvas.height = displayHeight
          const tempCtx = tempCanvas.getContext('2d')
          
          if (tempCtx) {
            // Draw mask scaled to display size
            tempCtx.drawImage(maskImg, 0, 0, displayWidth, displayHeight)
            
            // Get mask data
            const maskData = tempCtx.getImageData(0, 0, displayWidth, displayHeight)
            
            // Create overlay layer on a separate canvas
            const overlayCanvas = document.createElement('canvas')
            overlayCanvas.width = displayWidth
            overlayCanvas.height = displayHeight
            const overlayCtx = overlayCanvas.getContext('2d')
            
            if (overlayCtx) {
              // Create overlay with dark red tint for background (mask = 0)
              // Background areas (black in mask) should be tinted red
              // Foreground areas (white in mask) should remain transparent
              const overlay = overlayCtx.createImageData(displayWidth, displayHeight)
              
              for (let i = 0; i < maskData.data.length; i += 4) {
                const maskValue = maskData.data[i] / 255 // Use red channel as mask value (0-1)
                
                // If mask is dark (background), apply dark red tint
                if (maskValue < 0.5) {
                  // Dark red tint overlay
                  overlay.data[i] = 150     // R - dark red
                  overlay.data[i + 1] = 0   // G
                  overlay.data[i + 2] = 0   // B
                  overlay.data[i + 3] = 255 * opacity // A - use opacity slider
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
          
          setMaskLoaded(true)
        }
        
        maskImg.onerror = () => {
          setMaskLoaded(false)
        }
      } else {
        setMaskLoaded(false)
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
  }, [sessionId, image.id, result, maxHeight, opacity])
  
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

export default ImageWithMask

