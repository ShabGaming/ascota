import axios from 'axios'

const API_BASE_URL = '/api'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

export interface TypeRunOptions {
  enable_appendage_subtype: boolean
  resolution: 1000 | 1500 | 3000
}

export interface TypeResultItem {
  item_id: string
  label: string
  confidence: number
}

export const checkContextStatus = async (
  contextPath: string
): Promise<{ is_preprocessed: boolean }> => {
  const response = await apiClient.get('/sessions/check-context-status', {
    params: { context_path: contextPath },
  })
  return response.data
}

export interface PotteryGateStatus {
  complete: boolean
  total: number
  missing_count: number
  pottery_on_disk_count: number
  is_preprocessed?: boolean
}

export const checkPotteryGate = async (
  contextPath: string
): Promise<PotteryGateStatus> => {
  const response = await apiClient.get('/sessions/check-pottery-gate', {
    params: { context_path: contextPath },
  })
  return response.data
}

export interface SessionSummary {
  session_id: string
  context_path: string
  classification_type: string
  items_count: number
  results_count: number
  in_memory?: boolean
}

export const listSessions = async (): Promise<{ sessions: SessionSummary[] }> => {
  const response = await apiClient.get('/sessions/list')
  return response.data
}

export const loadSession = async (
  sessionId: string
): Promise<{
  session_id: string
  context_path: string
  classification_type: string
  items_count: number
  has_results: boolean
  options: Record<string, unknown>
}> => {
  const response = await apiClient.post('/sessions/load', { session_id: sessionId })
  return response.data
}

export const deleteSession = async (sessionId: string): Promise<void> => {
  await apiClient.delete(`/sessions/${sessionId}`)
}

export const createSession = async (
  contextPath: string,
  classificationType: string = 'type'
): Promise<{ session_id: string; items_count: number }> => {
  const response = await apiClient.post('/sessions', {
    context_path: contextPath,
    classification_type: classificationType,
  })
  return response.data
}

export const getSession = async (sessionId: string) => {
  const response = await apiClient.get(`/sessions/${sessionId}`)
  return response.data
}

export interface PotteryRunOptions {
  resolution: 1000 | 1500 | 3000
}

export interface PotteryResultRow {
  label: string
  confidence: number
  p_pottery?: number
}

export const runPotteryClassification = async (
  sessionId: string,
  options: PotteryRunOptions
): Promise<{ results: Array<{ item_id: string; label: string; confidence: number; p_pottery?: number }> }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/pottery/run`, options)
  return response.data
}

export const getPotteryResults = async (
  sessionId: string
): Promise<{ results: Record<string, PotteryResultRow> }> => {
  const response = await apiClient.get(`/sessions/${sessionId}/pottery/results`)
  return response.data
}

export const updatePotteryResult = async (
  sessionId: string,
  itemId: string,
  label: 'pottery' | 'non_pottery'
): Promise<void> => {
  await apiClient.patch(`/sessions/${sessionId}/pottery/results/${encodeURIComponent(itemId)}`, {
    label,
  })
}

export const undoPotteryEdit = async (
  sessionId: string
): Promise<{ restored: boolean; results?: Record<string, PotteryResultRow> }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/pottery/undo`)
  return response.data
}

export const runTypeClassification = async (
  sessionId: string,
  options: TypeRunOptions
): Promise<{ results: TypeResultItem[] }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/type/run`, options)
  return response.data
}

export const getTypeResults = async (
  sessionId: string
): Promise<{ results: Record<string, { label: string; confidence: number }> }> => {
  const response = await apiClient.get(`/sessions/${sessionId}/type/results`)
  return response.data
}

export const updateTypeResult = async (
  sessionId: string,
  itemId: string,
  label: string
): Promise<void> => {
  await apiClient.patch(`/sessions/${sessionId}/type/results/${encodeURIComponent(itemId)}`, {
    label,
  })
}

export const undoTypeEdit = async (
  sessionId: string
): Promise<{ restored: boolean; results?: Record<string, { label: string; confidence: number }> }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/type/undo`)
  return response.data
}

export const exportSession = async (
  sessionId: string
): Promise<{ saved_finds: number }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/export`)
  return response.data
}

export interface DecorationRunOptions {
  resolution: 1000 | 1500 | 3000
}

export const runDecorationClassification = async (
  sessionId: string,
  options: DecorationRunOptions
): Promise<{ results: TypeResultItem[] }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/decoration/run`, options)
  return response.data
}

export const getDecorationResults = async (
  sessionId: string
): Promise<{ results: Record<string, { label: string; confidence: number }> }> => {
  const response = await apiClient.get(`/sessions/${sessionId}/decoration/results`)
  return response.data
}

export const updateDecorationResult = async (
  sessionId: string,
  itemId: string,
  label: string
): Promise<void> => {
  await apiClient.patch(`/sessions/${sessionId}/decoration/results/${encodeURIComponent(itemId)}`, {
    label,
  })
}

export const undoDecorationEdit = async (
  sessionId: string
): Promise<{ restored: boolean; results?: Record<string, { label: string; confidence: number }> }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/decoration/undo`)
  return response.data
}

// Color clustering
export interface ColorRunResponse {
  results: Array<{ item_id: string; cluster_id: number }>
  clusters: Array<{ cluster_id: number; item_ids: string[] }>
  noise_item_ids: string[]
  cluster_names: Record<string, string>
}

export interface ColorResultsResponse {
  results: Record<string, { cluster_id: number }>
  cluster_names: Record<string, string>
  clusters: Array<{ cluster_id: number; item_ids: string[] }>
  noise_item_ids: string[]
}

export interface ReclusterOptions {
  min_cluster_size?: number
  min_samples?: number
  cluster_selection_epsilon?: number
  cluster_selection_method?: string
}

export interface ColorRunOptions {
  resolution?: 1000 | 1500 | 3000
}

export const runColorClustering = async (
  sessionId: string,
  options?: ColorRunOptions
): Promise<ColorRunResponse> => {
  const response = await apiClient.post(`/sessions/${sessionId}/color/run`, options ?? {})
  return response.data
}

export const getColorResults = async (
  sessionId: string
): Promise<ColorResultsResponse> => {
  const response = await apiClient.get(`/sessions/${sessionId}/color/results`)
  return response.data
}

export const reclusterColor = async (
  sessionId: string,
  options: ReclusterOptions
): Promise<ColorResultsResponse> => {
  const response = await apiClient.post(`/sessions/${sessionId}/color/recluster`, options)
  return response.data
}

export const updateColorResult = async (
  sessionId: string,
  itemId: string,
  clusterId: number
): Promise<void> => {
  await apiClient.patch(`/sessions/${sessionId}/color/results/${encodeURIComponent(itemId)}`, {
    cluster_id: clusterId,
  })
}

export const setColorClusterName = async (
  sessionId: string,
  clusterId: number,
  name: string
): Promise<void> => {
  await apiClient.patch(
    `/sessions/${sessionId}/color/cluster/${clusterId}/name`,
    { name }
  )
}

export const undoColorEdit = async (
  sessionId: string
): Promise<{ restored: boolean; results?: Record<string, { cluster_id: number }> }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/color/undo`)
  return response.data
}

// Texture clustering
export interface TextureRunResponse {
  results: Array<{ item_id: string; cluster_id: number }>
  clusters: Array<{ cluster_id: number; item_ids: string[] }>
  noise_item_ids: string[]
  cluster_names: Record<string, string>
}

export interface TextureResultsResponse {
  results: Record<string, { cluster_id: number }>
  cluster_names: Record<string, string>
  clusters: Array<{ cluster_id: number; item_ids: string[] }>
  noise_item_ids: string[]
}

export interface TextureRunOptions {
  resolution?: 1000 | 1500 | 3000
}

export const runTextureClustering = async (
  sessionId: string,
  options?: TextureRunOptions
): Promise<TextureRunResponse> => {
  const response = await apiClient.post(`/sessions/${sessionId}/texture/run`, options ?? {})
  return response.data
}

export const getTextureResults = async (sessionId: string): Promise<TextureResultsResponse> => {
  const response = await apiClient.get(`/sessions/${sessionId}/texture/results`)
  return response.data
}

export const reclusterTexture = async (
  sessionId: string,
  options: ReclusterOptions
): Promise<TextureResultsResponse> => {
  const response = await apiClient.post(`/sessions/${sessionId}/texture/recluster`, options)
  return response.data
}

export const updateTextureResult = async (
  sessionId: string,
  itemId: string,
  clusterId: number
): Promise<void> => {
  await apiClient.patch(`/sessions/${sessionId}/texture/results/${encodeURIComponent(itemId)}`, {
    cluster_id: clusterId,
  })
}

export const setTextureClusterName = async (
  sessionId: string,
  clusterId: number,
  name: string
): Promise<void> => {
  await apiClient.patch(
    `/sessions/${sessionId}/texture/cluster/${clusterId}/name`,
    { name }
  )
}

export const undoTextureEdit = async (
  sessionId: string
): Promise<{ restored: boolean; results?: Record<string, { cluster_id: number }> }> => {
  const response = await apiClient.post(`/sessions/${sessionId}/texture/undo`)
  return response.data
}

export interface ImageDisplayOptions {
  transparentMode?: boolean
  focusOnPottery?: boolean
}

export const getImageUrl = (
  sessionId: string,
  itemId: string,
  options?: ImageDisplayOptions
): string => {
  const base = `${API_BASE_URL}/sessions/${sessionId}/image/${encodeURIComponent(itemId)}`
  const transparent = options?.transparentMode ? 1 : 0
  const focus = options?.focusOnPottery ? 1 : 0
  if (transparent === 0 && focus === 0) return base
  const params = new URLSearchParams()
  if (transparent) params.set('transparent', '1')
  if (focus) params.set('focus', '1')
  return `${base}?${params.toString()}`
}
