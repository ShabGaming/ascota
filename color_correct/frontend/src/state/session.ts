import { create } from 'zustand'
import { ClusterResponse, ImageItem, Cluster } from '../api/client'

interface SessionState {
  sessionId: string | null
  clusters: Cluster[]
  images: Record<string, ImageItem>
  selectedImageId: string | null
  selectedClusterId: string | null
  
  setSession: (sessionId: string) => void
  setClusters: (data: ClusterResponse) => void
  updateClusterCorrection: (clusterId: string, cluster: Cluster) => void
  moveImageLocally: (imageId: string, targetClusterId: string) => void
  setSelectedImage: (imageId: string | null, clusterId: string | null) => void
  reset: () => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  clusters: [],
  images: {},
  selectedImageId: null,
  selectedClusterId: null,
  
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
    set({ selectedImageId: imageId, selectedClusterId: clusterId })
  },
  
  reset: () => {
    set({
      sessionId: null,
      clusters: [],
      images: {},
      selectedImageId: null,
      selectedClusterId: null
    })
  }
}))

