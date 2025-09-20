export interface PredictionResponse {
  ok?: boolean
  face_pred?: string
  face_scores?: number[]
  face_img?: string
  xray_pred?: string
  yolo_vis?: string
  found_labels?: string[]
  xray_img?: string
  combined?: string
  message?: string
}

export interface HealthResponse {
  status: string
  models?: {
    face?: boolean
    yolo?: boolean
  }
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
  const url = apiBase ? `${apiBase}/health` : '/health'
  
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

export async function postPredict(formData: FormData): Promise<PredictionResponse> {
  const apiBase = resolveApiBase()
  const url = apiBase ? `${apiBase}/predict` : '/predict'
  
  const response = await fetchWithTimeout(url, {
    method: 'POST',
    body: formData,
  }, 10000) // Longer timeout for file upload
  
  if (!response.ok) {
    let errorMessage = 'Analysis failed'
    try {
      const errorData = await response.json()
      errorMessage = errorData.message || errorMessage
    } catch {
      errorMessage = await response.text() || errorMessage
    }
    throw new Error(errorMessage)
  }
  
  return response.json()
}