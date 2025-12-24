import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Button,
  HStack,
  VStack,
  useToast,
  Select,
  FormControl,
  FormLabel,
} from '@chakra-ui/react'
import { useQuery, useMutation } from 'react-query'
import {
  getStage1Results,
  updateCards,
  addCard,
  deleteCard,
  getImageUrl,
  ImageItem,
  CardDetection,
} from '../api/client'
import { useSessionStore } from '../state/session'

interface CardEditorProps {
  sessionId: string
  imageId: string
  onClose: () => void
}

interface Point {
  x: number
  y: number
}

interface CardWithPoints extends CardDetection {
  points: Point[]
  isDragging?: number | null
}

function CardEditor({ sessionId, imageId, onClose }: CardEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [cards, setCards] = useState<CardWithPoints[]>([])
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [scale, setScale] = useState(1)
  const [selectedCardType, setSelectedCardType] = useState('24_color_card')
  const toast = useToast()
  
  const images = useSessionStore(state => state.images)
  const setStage1Results = useSessionStore(state => state.setStage1Results)
  const stage1Results = useSessionStore(state => state.stage1Results)
  
  // Load image and results
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
      if (canvas) {
        const maxWidth = 1200
        const maxHeight = 800
        const scaleX = maxWidth / img.width
        const scaleY = maxHeight / img.height
        const s = Math.min(scaleX, scaleY, 1)
        setScale(s)
        canvas.width = img.width * s
        canvas.height = img.height * s
      }
    }
  }, [imageId, images])
  
  // Load cards from results
  useEffect(() => {
    const result = stage1Results[imageId]
    if (result && result.cards) {
      const cardsWithPoints = result.cards.map(card => ({
        ...card,
        points: card.coordinates.map(coord => ({ x: coord[0] * scale, y: coord[1] * scale })),
        isDragging: null,
      }))
      setCards(cardsWithPoints)
    }
  }, [imageId, stage1Results, scale])
  
  // Draw function
  const draw = () => {
    const canvas = canvasRef.current
    if (!canvas || !image) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw image
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
    
    // Draw cards
    cards.forEach((card) => {
      const color = getCardColor(card.card_type)
      const rgb = getColorRGB(color)
      
      // Draw polygon
      ctx.strokeStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`
      ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.2)`
      ctx.lineWidth = 2
      
      ctx.beginPath()
      ctx.moveTo(card.points[0].x, card.points[0].y)
      for (let i = 1; i < card.points.length; i++) {
        ctx.lineTo(card.points[i].x, card.points[i].y)
      }
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
      
      // Draw points
      card.points.forEach((point, idx) => {
        ctx.fillStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`
        ctx.beginPath()
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2)
        ctx.fill()
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 2
        ctx.stroke()
      })
    })
  }
  
  useEffect(() => {
    draw()
  }, [cards, image, scale])
  
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
  
  const getPointAt = (x: number, y: number): { cardIndex: number; pointIndex: number } | null => {
    for (let i = 0; i < cards.length; i++) {
      for (let j = 0; j < cards[i].points.length; j++) {
        const point = cards[i].points[j]
        const dist = Math.sqrt((x - point.x) ** 2 + (y - point.y) ** 2)
        if (dist < 10) {
          return { cardIndex: i, pointIndex: j }
        }
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
    
    const hit = getPointAt(x, y)
    if (hit) {
      const newCards = [...cards]
      newCards[hit.cardIndex].isDragging = hit.pointIndex
      setCards(newCards)
    }
  }
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    const newCards = cards.map(card => {
      if (card.isDragging !== null && card.isDragging !== undefined) {
        const newPoints = [...card.points]
        newPoints[card.isDragging] = { x, y }
        return { ...card, points: newPoints }
      }
      return card
    })
    
    if (newCards.some(c => c.isDragging !== null)) {
      setCards(newCards)
    }
  }
  
  const handleMouseUp = () => {
    setCards(cards.map(card => ({ ...card, isDragging: null })))
  }
  
  const handleAddCard = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const size = 100
    
    const newCard: CardWithPoints = {
      card_id: `card_${Date.now()}`,
      card_type: selectedCardType,
      coordinates: [
        [centerX - size, centerY - size],
        [centerX + size, centerY - size],
        [centerX + size, centerY + size],
        [centerX - size, centerY + size],
      ],
      confidence: 1.0,
      points: [
        { x: centerX - size, y: centerY - size },
        { x: centerX + size, y: centerY - size },
        { x: centerX + size, y: centerY + size },
        { x: centerX - size, y: centerY + size },
      ],
      isDragging: null,
    }
    
    setCards([...cards, newCard])
  }
  
  const handleDeleteCard = (cardId: string) => {
    setCards(cards.filter(card => card.card_id !== cardId))
  }
  
  const updateMutation = useMutation(
    () => {
      const cardsToSave = cards.map(card => ({
        ...card,
        coordinates: card.points.map(p => [p.x / scale, p.y / scale]),
      }))
      return updateCards(sessionId, imageId, cardsToSave)
    },
    {
      onSuccess: () => {
        // Refresh results
        const result = stage1Results[imageId]
        if (result) {
          const updatedResult = {
            ...result,
            cards: cards.map(card => ({
              ...card,
              coordinates: card.points.map(p => [p.x / scale, p.y / scale]),
            })),
          }
          setStage1Results({
            ...stage1Results,
            [imageId]: updatedResult,
          })
        }
        toast({
          title: 'Cards updated',
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
        <FormControl>
          <FormLabel>Card Type</FormLabel>
          <Select
            value={selectedCardType}
            onChange={(e) => setSelectedCardType(e.target.value)}
          >
            <option value="24_color_card">24 Color Card</option>
            <option value="8_hybrid_card">8 Hybrid Card</option>
            <option value="checker_card">Checker Card</option>
          </Select>
        </FormControl>
        <Button onClick={handleAddCard} colorScheme="brand">
          Add Card
        </Button>
      </HStack>
      
      <Box border="1px" borderColor="gray.200" borderRadius="md" overflow="hidden">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ cursor: 'crosshair', display: 'block' }}
        />
      </Box>
      
      <VStack align="stretch" spacing={2}>
        {cards.map((card) => (
          <HStack key={card.card_id} justify="space-between" p={2} bg="gray.50" borderRadius="md">
            <Box>
              <Box fontWeight="bold" color={getCardColor(card.card_type)}>
                {card.card_type.replace('_', ' ')}
              </Box>
              <Box fontSize="sm" color="gray.600">
                Confidence: {(card.confidence * 100).toFixed(1)}%
              </Box>
            </Box>
            <Button
              size="sm"
              colorScheme="red"
              onClick={() => handleDeleteCard(card.card_id)}
            >
              Delete
            </Button>
          </HStack>
        ))}
      </VStack>
      
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

export default CardEditor

