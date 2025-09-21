// Enhanced structured response format with per-model and ROI details
export interface GenderPrediction {
  male: number
  female: number
  label: string
}

export interface EnsembleResult {
  method: string
  score: number
  models_used: number
  weights_used?: Record<string, number>
}

export interface Detection {
  box: number[] // [x1, y1, x2, y2]
  conf: number
  label: string
}

export interface ROIResult {
  roi_id: number
  box: number[]
  per_model: Record<string, number>
  ensemble: EnsembleResult
}

export interface ModalityResult {
  type: string
  label: string
  scores: number[]
  risk: string
  original_img?: string
  visualization?: string
  found_labels?: string[]
  
  // Face-specific fields
  gender?: GenderPrediction
  
  // X-ray-specific fields
  detections?: Detection[]
  per_roi?: ROIResult[]
  
  // Common fields
  per_model?: Record<string, number>
  ensemble?: EnsembleResult
}

export interface FinalResult {
  risk: string
  confidence: number
  explanation: string
  fusion_mode: string
}

export interface StructuredPredictionResponse {
  ok: boolean
  modalities: ModalityResult[]
  final: FinalResult
  warnings: string[]
  processing_time_ms: number
  debug: Record<string, any>
}

// Legacy response format for backward compatibility
export interface LegacyPredictionResponse {
  ok?: boolean
  face_pred?: string
  face_scores?: number[]
  face_img?: string
  face_risk?: string
  xray_pred?: string
  yolo_vis?: string
  found_labels?: string[]
  xray_img?: string
  xray_risk?: string
  combined?: string
  overall_risk?: string
  message?: string
}

// Union type for responses
export type PredictionResponse = StructuredPredictionResponse | LegacyPredictionResponse

export interface ModelStatus {
  status: string
  file_exists: boolean
  lazy_loadable: boolean
  path?: string
  error?: string
  version?: string
}

export interface EnhancedHealthResponse {
  status: string
  models: Record<string, ModelStatus>
  uptime_seconds: number
  version: string
}

export type HealthStatus = 'online' | 'offline' | 'unreachable'

export function resolveApiBase(): string {
  // 1. Check query param first
  const params = new URLSearchParams(window.location.search)
  const apiParam = params.get('api')
  if (apiParam) {
    return apiParam.replace(/\/+$/, '') // Remove trailing slashes
  }
  
  // 2. Check environment variable
  const envBase = import.meta.env.VITE_API_BASE
  if (envBase) {
    return envBase.replace(/\/+$/, '')
  }
  
  // 3. Fall back to relative path (Vite proxy)
  return ''
}

export function withBase(path?: string): string {
  if (!path) return ''
  const apiBase = resolveApiBase()
  if (!apiBase) return path
  return `${apiBase}${path}`
}

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 5000): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    clearTimeout(timeoutId)
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    throw error
  }
}

export async function pingHealth(): Promise<HealthStatus> {
  const apiBase = resolveApiBase()
  const url = withBase('/health')
  
  try {
    const response = await fetchWithTimeout(url, {}, 5000)
    
    if (response.ok) {
      return 'online'
    } else if (response.status >= 500) {
      return 'offline'
    } else {
      return 'unreachable'
    }
  } catch (error) {
    // Network errors, CORS, timeout, etc.
    return 'unreachable'
  }
}

export async function postPredict(formData: FormData, useStructured: boolean = true): Promise<PredictionResponse> {
  const url = withBase(useStructured ? '/predict' : '/predict-legacy')
  
  const response = await fetchWithTimeout(url, {
    method: 'POST',
    body: formData,
  }, 10000) // Longer timeout for file upload
  
  if (!response.ok) {
    let errorMessage = 'Analysis failed'
    try {
      const errorData = await response.json()
      if (errorData.ok === false) {
        errorMessage = errorData.details || errorMessage
      } else {
        errorMessage = errorData.message || errorData.detail || errorMessage
      }
    } catch {
      errorMessage = await response.text() || errorMessage
    }
    throw new Error(errorMessage)
  }
  
  return response.json()
}

export async function postPredictFile(file: File, type: 'face' | 'xray'): Promise<any> {
  const formData = new FormData()
  formData.append('file', file)
  
  const url = withBase(`/predict-file?type=${type}`)
  
  const response = await fetchWithTimeout(url, {
    method: 'POST',
    body: formData,
  }, 10000)
  
  if (!response.ok) {
    let errorMessage = 'Analysis failed'
    try {
      const errorData = await response.json()
      errorMessage = errorData.details || errorData.message || errorData.detail || errorMessage
    } catch {
      errorMessage = await response.text() || errorMessage
    }
    throw new Error(errorMessage)
  }
  
  return response.json()
}
export async function postPredictFile(file: File, type: 'face' | 'xray'): Promise<any> {
  const formData = new FormData()
  formData.append('file', file)
  
  const url = withBase(`/predict-file?type=${type}`)
  
  const response = await fetchWithTimeout(url, {
    method: 'POST',
    body: formData,
  }, 10000)
  
  if (!response.ok) {
    let errorMessage = 'Analysis failed'
    try {
      const errorData = await response.json()
      errorMessage = errorData.message || errorData.detail || errorMessage
    } catch {
      errorMessage = await response.text() || errorMessage
    }
    throw new Error(errorMessage)
  }
  
  return response.json()
}

export async function getEnhancedHealth(): Promise<EnhancedHealthResponse> {
  const url = withBase('/health')
  
  const response = await fetchWithTimeout(url, {}, 5000)
  
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`)
  }
  
  return response.json()
}

// Helper function to check if response is structured format
export function isStructuredResponse(response: PredictionResponse): response is StructuredPredictionResponse {
  return 'modalities' in response && 'final' in response
}

// Helper function to convert structured to legacy format
export function convertToLegacyFormat(structured: StructuredPredictionResponse): LegacyPredictionResponse {
  const legacy: LegacyPredictionResponse = { ok: structured.ok }
  
  // Extract face data
  const faceModality = structured.modalities.find(m => m.type === 'face')
  if (faceModality) {
    legacy.face_pred = faceModality.label
    legacy.face_scores = faceModality.scores
    legacy.face_img = faceModality.original_img
    legacy.face_risk = faceModality.risk
  }
  
  // Extract X-ray data
  const xrayModality = structured.modalities.find(m => m.type === 'xray')
  if (xrayModality) {
    legacy.xray_pred = xrayModality.label
    legacy.xray_img = xrayModality.original_img
    legacy.yolo_vis = xrayModality.visualization
    legacy.found_labels = xrayModality.found_labels
    legacy.xray_risk = xrayModality.risk
  }
  
  // Final results
  legacy.combined = structured.final.explanation
  legacy.overall_risk = structured.final.risk
  legacy.message = 'ok'
  
  return legacy
}