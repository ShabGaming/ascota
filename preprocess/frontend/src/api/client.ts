import axios from 'axios'

const API_BASE_URL = '/api'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface ImageItem {
  id: string
  context_id: string
  find_number: string
  raw_path?: string
  proxy_3000?: string
  proxy_1500?: string
  proxy_450?: string
  primary_path: string
}

export interface CardDetection {
  card_id: string
  card_type: string
  coordinates: number[][]
  confidence: number
}

export interface ImageCardResult {
  image_id: string
  image_size: number[]
  cards: CardDetection[]
  error?: string
}

export interface Stage1Results {
  results: Record<string, ImageCardResult>
}

export interface MaskResult {
  image_id: string
  mask_path?: string
  error?: string
}

export interface Stage2Results {
  results: Record<string, MaskResult>
}

export interface ScaleResult {
  image_id: string
  pixels_per_cm?: number
  surface_area_cm2?: number
  method?: string
  card_used?: string
  centers?: number[][]  // Circle centers for 8_hybrid_card in original image coordinates
  error?: string
}

export interface Stage3Results {
  results: Record<string, ScaleResult>
}

// Session management
export interface CreateSessionRequest {
  contexts: string[]
  overwrite_existing?: boolean
}

export interface CreateSessionResponse {
  session_id: string
}

export const createSession = async (request: CreateSessionRequest): Promise<CreateSessionResponse> => {
  const response = await apiClient.post('/sessions', request)
  return response.data
}

export const listSessions = async (): Promise<{ session_ids: string[] }> => {
  const response = await apiClient.get('/sessions')
  return response.data
}

export const deleteSession = async (sessionId: string): Promise<void> => {
  await apiClient.delete(`/sessions/${sessionId}`)
}

export const checkContextStatus = async (contextPath: string): Promise<{ is_preprocessed: boolean }> => {
  const response = await apiClient.get('/sessions/check-context-status', {
    params: { context_path: contextPath }
  })
  return response.data
}

export const getSessionImages = async (sessionId: string): Promise<{
  images: Record<string, ImageItem>
  total: number
}> => {
  const response = await apiClient.get(`/sessions/${sessionId}/images`)
  return response.data
}

// Stage 1: Card Detection
export const detectCards = async (sessionId: string): Promise<Stage1Results> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage1/detect`)
  return response.data
}

export const getStage1Results = async (sessionId: string): Promise<Stage1Results> => {
  const response = await apiClient.get(`/sessions/${sessionId}/stage1/results`)
  return response.data
}

export const updateCards = async (
  sessionId: string,
  imageId: string,
  cards: CardDetection[]
): Promise<void> => {
  await apiClient.put(`/sessions/${sessionId}/stage1/image/${imageId}/cards`, { cards })
}

export const addCard = async (
  sessionId: string,
  imageId: string,
  card: CardDetection
): Promise<CardDetection> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage1/image/${imageId}/cards`, card)
  return response.data.card
}

export const deleteCard = async (
  sessionId: string,
  imageId: string,
  cardId: string
): Promise<void> => {
  await apiClient.delete(`/sessions/${sessionId}/stage1/image/${imageId}/cards/${cardId}`)
}

export const saveStage1 = async (sessionId: string): Promise<{ message: string; saved_count: number }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage1/save`)
  return response.data
}

// Stage 2: Mask Generation
export const generateMasks = async (sessionId: string): Promise<Stage2Results> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage2/generate`)
  return response.data
}

export const getStage2Results = async (sessionId: string): Promise<Stage2Results> => {
  const response = await apiClient.get(`/sessions/${sessionId}/stage2/results`)
  return response.data
}

export const updateMask = async (
  sessionId: string,
  imageId: string,
  maskData: string  // base64 encoded PNG
): Promise<{ message: string; mask_path: string }> => {
  const response = await apiClient.put(`/sessions/${sessionId}/stage2/image/${imageId}/mask`, {
    mask_data: maskData
  })
  return response.data
}

export const saveStage2 = async (sessionId: string): Promise<{ message: string; saved_count: number }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage2/save`)
  return response.data
}

export const wandSelect = async (
  sessionId: string,
  imageId: string,
  x: number,
  y: number
): Promise<{ mask_data: string }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage2/image/${imageId}/wand-select`, {
    x,
    y
  })
  return response.data
}

// Stage 3: Scale Calculation
export const calculateScale = async (sessionId: string): Promise<Stage3Results> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage3/calculate`)
  return response.data
}

export const getStage3Results = async (sessionId: string): Promise<Stage3Results> => {
  const response = await apiClient.get(`/sessions/${sessionId}/stage3/results`)
  return response.data
}

export const updateCenters = async (
  sessionId: string,
  imageId: string,
  centers: number[][]
): Promise<{ message: string; pixels_per_cm: number; surface_area_cm2?: number }> => {
  const response = await apiClient.put(`/sessions/${sessionId}/stage3/image/${imageId}/centers`, {
    centers
  })
  return response.data
}

export const calculateSurfaceArea = async (
  sessionId: string,
  imageId: string
): Promise<{ message: string; surface_area_cm2: number }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage3/image/${imageId}/surface_area`)
  return response.data
}

export const saveStage3 = async (sessionId: string): Promise<{ message: string; saved_count: number }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/stage3/save`)
  return response.data
}

// Image serving
export const getImageUrl = (sessionId: string, imageId: string, size: '3000' | '1500' | '450' = '3000'): string => {
  return `${API_BASE_URL}/sessions/${sessionId}/image/${imageId}?size=${size}`
}

export const getCardCropUrl = (sessionId: string, imageId: string): string => {
  return `${API_BASE_URL}/sessions/${sessionId}/card_crop/${imageId}`
}

export const getMaskUrl = (sessionId: string, imageId: string, maskPath: string, cacheBust?: number): string => {
  // Construct URL to serve mask from .ascota folder
  // Add cache-busting parameter to force browser to reload when mask is updated
  const timestamp = cacheBust || Date.now()
  return `${API_BASE_URL}/sessions/${sessionId}/masks/${imageId}?path=${encodeURIComponent(maskPath)}&t=${timestamp}`
}

