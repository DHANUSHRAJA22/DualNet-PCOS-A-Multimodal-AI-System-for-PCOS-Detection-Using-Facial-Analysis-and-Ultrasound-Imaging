import { useState, useRef, useCallback, useEffect } from 'react'
import { Camera, X, Check, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { toast } from 'sonner'

interface CameraCaptureProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCapture: (file: File) => void
  facingMode?: 'user' | 'environment'
  quality?: number
  maxSize?: { width?: number; height?: number }
}

export function CameraCapture({
  open,
  onOpenChange,
  onCapture,
  facingMode = 'user',
  quality = 0.92,
  maxSize,
}: CameraCaptureProps) {
  // Keep MediaStream in a ref to avoid render loops that cause flicker
  const streamRef = useRef<MediaStream | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [ready, setReady] = useState(false)     // video metadata loaded
  const [captured, setCaptured] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Prevent double starts (React 18 StrictMode, rapid re-opens, etc.)
  const startingRef = useRef(false)

  const attachVideo = useCallback(async () => {
    const video = videoRef.current
    const stream = streamRef.current
    if (!video || !stream) return
    // If already attached, do nothing
    if ((video as any).srcObject === stream) return

    ;(video as any).srcObject = stream
    // Wait for metadata before showing to avoid black-frame flicker
    const onMeta = () => {
      setReady(true)
      video.play().catch(() => {})
    }
    if (video.readyState >= 1) {
      onMeta()
    } else {
      video.addEventListener('loadedmetadata', onMeta, { once: true })
    }
  }, [])

  const startCamera = useCallback(async () => {
    if (startingRef.current || streamRef.current) return
    try {
      startingRef.current = true
      setIsLoading(true)
      setError(null)
      setReady(false)

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera API not available in this browser.')
      }
      if (!window.isSecureContext) {
        throw new Error('Camera requires HTTPS or localhost.')
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode },
        audio: false,
      })
      streamRef.current = stream
      await attachVideo()
    } catch (err: any) {
      console.error('Camera access error:', err)
      const name = err?.name as string | undefined
      if (name === 'NotAllowedError') {
        setError('Camera permission denied. Allow access and try again.')
      } else if (name === 'NotFoundError') {
        setError('No camera found. Connect a camera and try again.')
      } else {
        setError(err?.message ?? 'Unable to access camera.')
      }
      toast.error('Camera access failed')
    } finally {
      setIsLoading(false)
      startingRef.current = false
    }
  }, [attachVideo, facingMode])

  const stopCamera = useCallback(() => {
    const s = streamRef.current
    if (s) {
      s.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    setCaptured(null)
    setReady(false)
    setError(null)
    if (videoRef.current) {
      ;(videoRef.current as any).srcObject = null
    }
  }, [])

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !ready) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let w = video.videoWidth
    let h = video.videoHeight
    if (maxSize && (maxSize.width || maxSize.height)) {
      const ar = w / h
      if (maxSize.width && (!maxSize.height || maxSize.width / ar <= (maxSize.height ?? Infinity))) {
        w = Math.min(w, maxSize.width)
        h = Math.round(w / ar)
      } else if (maxSize.height) {
        h = Math.min(h, maxSize.height)
        w = Math.round(h * ar)
      }
    }

    canvas.width = w
    canvas.height = h
    ctx.drawImage(video, 0, 0, w, h)
    setCaptured(canvas.toDataURL('image/jpeg', quality))
  }, [maxSize, quality, ready])

  const savePhoto = useCallback(() => {
    if (!canvasRef.current) return
    canvasRef.current.toBlob(
      (blob) => {
        if (!blob) return
        if (blob.size > 5 * 1024 * 1024) {
          toast.error(`Captured image too large (${(blob.size / 1024 / 1024).toFixed(1)}MB). Max is 5MB.`)
          return
        }
        const file = new File([blob], `capture-${Date.now()}.jpg`, {
          type: 'image/jpeg',
          lastModified: Date.now(),
        })
        onCapture(file)
        onOpenChange(false)
        stopCamera()
      },
      'image/jpeg',
      quality
    )
  }, [onCapture, onOpenChange, stopCamera, quality])

  const retakePhoto = useCallback(() => setCaptured(null), [])

  // Open/close lifecycle (only toggles; never calls start twice)
  useEffect(() => {
    if (open) startCamera()
    else stopCamera()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // Re-attach stream on re-renders if the video element remounts
  useEffect(() => {
    if (open && streamRef.current) attachVideo()
  }, [open, attachVideo])

  // Cleanup on unmount
  useEffect(() => {
    return () => stopCamera()
  }, [stopCamera])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Camera Capture
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
            {/* Overlays */}
            {error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-white p-4">
                <AlertCircle className="h-12 w-12 mb-4 text-red-400" />
                <p className="text-center text-sm">{error}</p>
                <Button onClick={startCamera} variant="outline" className="mt-4 bg-white/10 border-white/20 text-white hover:bg-white/20">
                  Try Again
                </Button>
              </div>
            )}
            {!error && !ready && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-white p-4 gap-3">
                {isLoading ? 'Starting camera…' : (
                  <>
                    <div>Camera access required</div>
                    <Button onClick={startCamera} variant="secondary">
                      <Camera className="h-4 w-4 mr-2" />
                      Start camera
                    </Button>
                  </>
                )}
              </div>
            )}

            {/* Live preview (hidden until ready to avoid flicker) */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover transition-opacity duration-200 ${ready ? 'opacity-100' : 'opacity-0'}`}
            />
            {/* Captured preview */}
            {captured && (
              <img src={captured} alt="Captured" className="w-full h-full object-cover" />
            )}

            <canvas ref={canvasRef} className="hidden" />
          </div>

          <div className="flex justify-center gap-3">
            {ready && !captured && !error && (
              <Button onClick={capturePhoto} size="lg">
                <Camera className="h-4 w-4 mr-2" />
                Capture Photo
              </Button>
            )}
            {captured && (
              <>
                <Button onClick={retakePhoto} variant="outline">
                  <X className="h-4 w-4 mr-2" />
                  Retake
                </Button>
                <Button onClick={savePhoto}>
                  <Check className="h-4 w-4 mr-2" />
                  Use Photo
                </Button>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
