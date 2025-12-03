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
  preview_resolution?: number
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

// Stable hash function for correction params (faster than JSON.stringify)
export const hashCorrectionParams = (params?: CorrectionParams): string => {
  if (!params) return 'none'
  // Round to 2 decimal places to avoid floating point precision issues
  return `${params.temperature.toFixed(0)}_${params.tint.toFixed(0)}_${params.exposure.toFixed(2)}_${params.contrast.toFixed(2)}_${params.saturation.toFixed(2)}_${params.red_gain.toFixed(2)}_${params.green_gain.toFixed(2)}_${params.blue_gain.toFixed(2)}`
}

export const getPreviewUrl = (
  sessionId: string,
  imageId: string,
  clusterId?: string,
  maxSize: number = 600,
  overallParams?: CorrectionParams,
  individualParams?: CorrectionParams,
  showOverall: boolean = true,
  showIndividual: boolean = true
): string => {
  const params = new URLSearchParams({
    image_id: imageId,
    max_size: maxSize.toString(),
    show_overall: showOverall.toString(),
    show_individual: showIndividual.toString(),
  })
  if (clusterId) {
    params.append('cluster_id', clusterId)
  }
  // Add overall correction parameters for real-time preview
  if (overallParams) {
    params.append('temperature', overallParams.temperature.toString())
    params.append('tint', overallParams.tint.toString())
    params.append('exposure', overallParams.exposure.toString())
    params.append('contrast', overallParams.contrast.toString())
    params.append('saturation', overallParams.saturation.toString())
    params.append('red_gain', overallParams.red_gain.toString())
    params.append('green_gain', overallParams.green_gain.toString())
    params.append('blue_gain', overallParams.blue_gain.toString())
  }
  // Add individual correction parameters if provided
  if (individualParams) {
    // Use a prefix to distinguish individual params
    params.append('individual_temperature', individualParams.temperature.toString())
    params.append('individual_tint', individualParams.tint.toString())
    params.append('individual_exposure', individualParams.exposure.toString())
    params.append('individual_contrast', individualParams.contrast.toString())
    params.append('individual_saturation', individualParams.saturation.toString())
    params.append('individual_red_gain', individualParams.red_gain.toString())
    params.append('individual_green_gain', individualParams.green_gain.toString())
    params.append('individual_blue_gain', individualParams.blue_gain.toString())
  }
  return `${API_BASE_URL}/sessions/${sessionId}/preview?${params.toString()}`
}

export const startExport = async (sessionId: string): Promise<ExportResponse> => {
  const response = await apiClient.post(`/sessions/${sessionId}/export`)
  return response.data
}

export interface SessionInfo {
  session_id: string
  saved_at: string
  context_count: number
  cluster_count: number
  image_count: number
  contexts: string[]
}

export interface ListSessionsResponse {
  active_sessions: string[]
  persisted_sessions: SessionInfo[]
}

export const listSessions = async (): Promise<ListSessionsResponse> => {
  const response = await apiClient.get('/sessions')
  return response.data
}

export const restoreSession = async (sessionId: string): Promise<{ session_id: string; message: string }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/restore`)
  return response.data
}

export const deleteSession = async (sessionId: string): Promise<void> => {
  await apiClient.delete(`/sessions/${sessionId}`)
}

export const checkContextStatus = async (contextPath: string): Promise<{ is_color_corrected: boolean }> => {
  const response = await apiClient.get('/sessions/check-context-status', {
    params: { context_path: contextPath }
  })
  return response.data
}

export const setIndividualCorrection = async (
  sessionId: string,
  imageId: string,
  params: CorrectionParams
): Promise<void> => {
  await apiClient.post(`/sessions/${sessionId}/images/${imageId}/set-individual-correction`, {
    params
  })
}

export const resetIndividualCorrections = async (
  sessionId: string,
  clusterId: string
): Promise<{ message: string }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/clusters/${clusterId}/reset-individual`)
  return response.data
}

