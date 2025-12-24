import { create } from 'zustand'
import { ImageItem, ImageCardResult, MaskResult, ScaleResult } from '../api/client'

type Stage = 1 | 2 | 3

interface SessionState {
  sessionId: string | null
  currentStage: Stage
  images: Record<string, ImageItem>
  stage1Results: Record<string, ImageCardResult>
  stage2Results: Record<string, MaskResult>
  stage3Results: Record<string, ScaleResult>
  selectedImageId: string | null
  
  setSession: (sessionId: string) => void
  setCurrentStage: (stage: Stage) => void
  setImages: (images: Record<string, ImageItem>) => void
  setStage1Results: (results: Record<string, ImageCardResult>) => void
  setStage2Results: (results: Record<string, MaskResult>) => void
  setStage3Results: (results: Record<string, ScaleResult>) => void
  setSelectedImage: (imageId: string | null) => void
  reset: () => void
}

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: null,
  currentStage: 1,
  images: {},
  stage1Results: {},
  stage2Results: {},
  stage3Results: {},
  selectedImageId: null,
  
  setSession: (sessionId: string) => {
    set({ sessionId })
  },
  
  setCurrentStage: (stage: Stage) => {
    set({ currentStage: stage })
  },
  
  setImages: (images: Record<string, ImageItem>) => {
    set({ images })
  },
  
  setStage1Results: (results: Record<string, ImageCardResult>) => {
    set({ stage1Results: results })
  },
  
  setStage2Results: (results: Record<string, MaskResult>) => {
    set({ stage2Results: results })
  },
  
  setStage3Results: (results: Record<string, ScaleResult>) => {
    set({ stage3Results: results })
  },
  
  setSelectedImage: (imageId: string | null) => {
    set({ selectedImageId: imageId })
  },
  
  reset: () => {
    set({
      sessionId: null,
      currentStage: 1,
      images: {},
      stage1Results: {},
      stage2Results: {},
      stage3Results: {},
      selectedImageId: null
    })
  }
}))

