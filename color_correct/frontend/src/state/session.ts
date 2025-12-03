import { create } from 'zustand'
import { ClusterResponse, ImageItem, Cluster, CorrectionParams } from '../api/client'

interface SessionState {
  sessionId: string | null
  clusters: Cluster[]
  images: Record<string, ImageItem>
  selectedImageId: string | null
  selectedClusterId: string | null
  pendingOverallCorrection: Record<string, CorrectionParams | null> // clusterId -> pending params
  pendingIndividualCorrection: Record<string, CorrectionParams | null> // imageId -> pending params
  
  setSession: (sessionId: string) => void
  setClusters: (data: ClusterResponse) => void
  updateClusterCorrection: (clusterId: string, cluster: Cluster) => void
  moveImageLocally: (imageId: string, targetClusterId: string) => void
  setSelectedImage: (imageId: string | null, clusterId: string | null) => void
  setPendingOverallCorrection: (clusterId: string, params: CorrectionParams | null) => void
  setPendingIndividualCorrection: (imageId: string, params: CorrectionParams | null) => void
  clearPendingCorrections: () => void
  reset: () => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  clusters: [],
  images: {},
  selectedImageId: null,
  selectedClusterId: null,
  pendingOverallCorrection: {},
  pendingIndividualCorrection: {},
  
  setSession: (sessionId: string) => {
    set({ sessionId })
  },
  
  setClusters: (data: ClusterResponse) => {
    set({
      clusters: data.clusters,
      images: data.images
    })
  },
  
  updateClusterCorrection: (clusterId: string, cluster: Cluster) => {
    set(state => ({
      clusters: state.clusters.map(c => 
        c.id === clusterId ? cluster : c
      )
    }))
  },
  
  moveImageLocally: (imageId: string, targetClusterId: string) => {
    set(state => {
      const newClusters = state.clusters.map(cluster => ({
        ...cluster,
        image_ids: cluster.image_ids.filter(id => id !== imageId)
      }))
      
      const targetIndex = newClusters.findIndex(c => c.id === targetClusterId)
      if (targetIndex !== -1) {
        newClusters[targetIndex].image_ids.push(imageId)
      }
      
      return { clusters: newClusters }
    })
  },
  
  setSelectedImage: (imageId: string | null, clusterId: string | null) => {
    const state = get()
    // Only clear pending corrections when switching to a different image or closing
    // Don't clear if selecting the same image (prevents unnecessary re-renders)
    const isSwitchingImage = state.selectedImageId !== imageId
    const isClosing = imageId === null
    
    set({ 
      selectedImageId: imageId, 
      selectedClusterId: clusterId,
      // Only clear pending corrections when actually switching images or closing
      ...(isSwitchingImage || isClosing ? {
        pendingOverallCorrection: {},
        pendingIndividualCorrection: {}
      } : {})
    })
  },
  
  setPendingOverallCorrection: (clusterId: string, params: CorrectionParams | null) => {
    set(state => ({
      pendingOverallCorrection: {
        ...state.pendingOverallCorrection,
        [clusterId]: params
      }
    }))
  },
  
  setPendingIndividualCorrection: (imageId: string, params: CorrectionParams | null) => {
    set(state => ({
      pendingIndividualCorrection: {
        ...state.pendingIndividualCorrection,
        [imageId]: params
      }
    }))
  },
  
  clearPendingCorrections: () => {
    set({
      pendingOverallCorrection: {},
      pendingIndividualCorrection: {}
    })
  },
  
  reset: () => {
    set({
      sessionId: null,
      clusters: [],
      images: {},
      selectedImageId: null,
      selectedClusterId: null,
      pendingOverallCorrection: {},
      pendingIndividualCorrection: {}
    })
  }
}))

