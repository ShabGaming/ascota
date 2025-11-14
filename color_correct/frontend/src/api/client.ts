import axios from 'axios'

const API_BASE_URL = '/api'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export type ImageSourceMode = "450px" | "1500px" | "3000px" | "raw_mode"

export interface SessionOptions {
  image_source: ImageSourceMode
  overwrite: boolean
  custom_k?: number
  sensitivity: number
}

export interface CreateSessionRequest {
  contexts: string[]
  options: SessionOptions
}

export interface CreateSessionResponse {
  session_id: string
}

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

export interface CorrectionParams {
  temperature: number
  tint: number
  exposure: number
  contrast: number
  saturation: number
  red_gain: number
  green_gain: number
  blue_gain: number
}

export interface Cluster {
  id: string
  image_ids: string[]
  correction_params?: CorrectionParams
}

export interface ClusterResponse {
  clusters: Cluster[]
  images: Record<string, ImageItem>
}

export interface JobStatusResponse {
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message?: string
  result?: any
  error?: string
}

export interface ExportSummary {
  total_images: number
  total_files_written: number
  overwritten_count: number
  new_files_count: number
  failed_count: number
  errors: string[]
}

export interface ExportResponse {
  job_id: string
  summary?: ExportSummary
}

// API functions
export const createSession = async (request: CreateSessionRequest): Promise<CreateSessionResponse> => {
  const response = await apiClient.post('/sessions', request)
  return response.data
}

export const startClustering = async (
  sessionId: string,
  sensitivity?: number
): Promise<{ job_id: string }> => {
  const url = `/sessions/${sessionId}/cluster${sensitivity !== undefined ? `?sensitivity=${sensitivity}` : ''}`
  const response = await apiClient.post(url)
  return response.data
}

export const undoClusters = async (sessionId: string): Promise<void> => {
  await apiClient.post(`/sessions/${sessionId}/clusters/undo`)
}

export const getJobStatus = async (sessionId: string, jobId: string): Promise<JobStatusResponse> => {
  const response = await apiClient.get(`/sessions/${sessionId}/status`, {
    params: { job_id: jobId }
  })
  return response.data
}

export const getSessionInfo = async (sessionId: string): Promise<{
  session_id: string
  contexts: string[]
  options: SessionOptions
}> => {
  const response = await apiClient.get(`/sessions/${sessionId}`)
  return response.data
}

export const getDiscoveredImages = async (sessionId: string): Promise<{
  images: Record<string, ImageItem>
  total: number
  image_source: string
}> => {
  const response = await apiClient.get(`/sessions/${sessionId}/discovered-images`)
  return response.data
}

export const createCluster = async (
  sessionId: string,
  imageId?: string
): Promise<{ cluster_id: string }> => {
  const params = imageId ? `?image_id=${encodeURIComponent(imageId)}` : ''
  const response = await apiClient.post(`/sessions/${sessionId}/clusters${params}`)
  return response.data
}

export const deleteCluster = async (
  sessionId: string,
  clusterId: string
): Promise<void> => {
  await apiClient.delete(`/sessions/${sessionId}/clusters/${clusterId}`)
}

export const getClusters = async (sessionId: string): Promise<ClusterResponse> => {
  const response = await apiClient.get(`/sessions/${sessionId}/clusters`)
  return response.data
}

export const moveImage = async (
  sessionId: string,
  imageId: string,
  targetClusterId: string
): Promise<void> => {
  await apiClient.patch(`/sessions/${sessionId}/clusters/move`, {
    image_id: imageId,
    target_cluster_id: targetClusterId
  })
}

export const autoCorrectCluster = async (
  sessionId: string,
  clusterId: string,
  imageId?: string
): Promise<{ params: CorrectionParams }> => {
  const url = `/sessions/${sessionId}/clusters/${clusterId}/auto-correct${imageId ? `?image_id=${encodeURIComponent(imageId)}` : ''}`
  const response = await apiClient.post(url)
  return response.data
}

export const setClusterCorrection = async (
  sessionId: string,
  clusterId: string,
  params: CorrectionParams
): Promise<void> => {
  await apiClient.post(`/sessions/${sessionId}/clusters/${clusterId}/set-correction`, {
    params
  })
}

export const resetClusterCorrection = async (
  sessionId: string,
  clusterId: string
): Promise<{ params: CorrectionParams }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/clusters/${clusterId}/reset`)
  return response.data
}

export const getPreviewUrl = (
  sessionId: string,
  imageId: string,
  clusterId?: string,
  maxSize: number = 600,
  correctionParams?: CorrectionParams
): string => {
  const params = new URLSearchParams({
    image_id: imageId,
    max_size: maxSize.toString(),
  })
  if (clusterId) {
    params.append('cluster_id', clusterId)
  }
  // Add correction parameters for real-time preview
  if (correctionParams) {
    params.append('temperature', correctionParams.temperature.toString())
    params.append('tint', correctionParams.tint.toString())
    params.append('exposure', correctionParams.exposure.toString())
    params.append('contrast', correctionParams.contrast.toString())
    params.append('saturation', correctionParams.saturation.toString())
    params.append('red_gain', correctionParams.red_gain.toString())
    params.append('green_gain', correctionParams.green_gain.toString())
    params.append('blue_gain', correctionParams.blue_gain.toString())
  }
  return `${API_BASE_URL}/sessions/${sessionId}/preview?${params.toString()}`
}

export const startExport = async (sessionId: string): Promise<ExportResponse> => {
  const response = await apiClient.post(`/sessions/${sessionId}/export`)
  return response.data
}

