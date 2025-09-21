import { useState, useRef, useCallback, useEffect } from 'react'
import { Camera, X, Check, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { toast } from 'sonner'

interface CameraCaptureProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCapture: (file: File) => void
}

export function CameraCapture({ open, onOpenChange, onCapture }: CameraCaptureProps) {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [captured, setCaptured] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false,
      })
      
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (error) {
      console.error('Camera access error:', error)
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          setError('Camera permission denied. Please allow camera access and try again.')
        } else if (error.name === 'NotFoundError') {
          setError('No camera found. Please connect a camera and try again.')
        } else {
          setError('Unable to access camera. Please check permissions and try again.')
        }
      } else {
        setError('Unable to access camera. Please check permissions and try again.')
      }
      toast.error('Camera access failed')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setCaptured(null)
    setError(null)
  }, [stream])

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    if (!ctx) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    const capturedImageUrl = canvas.toDataURL('image/jpeg', 0.92)
    setCaptured(capturedImageUrl)
  }, [])

  const savePhoto = useCallback(() => {
    if (!canvasRef.current) return

    canvasRef.current.toBlob((blob) => {
      if (blob) {
        // Validate blob size (5MB limit)
        const maxSize = 5 * 1024 * 1024 // 5MB
        if (blob.size > maxSize) {
          toast.error(`Captured image is too large (${(blob.size / 1024 / 1024).toFixed(1)}MB). Maximum size is 5MB.`)
          return
        }
        
        const file = new File([blob], `capture-${Date.now()}.jpg`, {
          type: 'image/jpeg',
          lastModified: Date.now(),
        })
        onCapture(file)
        onOpenChange(false)
        stopCamera()
      }
    }, 'image/jpeg', 0.92)
  }, [onCapture, onOpenChange, stopCamera])

  const retakePhoto = useCallback(() => {
    setCaptured(null)
  }, [])

  const handleOpenChange = useCallback((newOpen: boolean) => {
    if (newOpen) {
      startCamera()
    } else {
      stopCamera()
    }
    onOpenChange(newOpen)
  }, [startCamera, stopCamera, onOpenChange])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [stream])

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Camera Capture
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
            {error ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-white p-4">
                <AlertCircle className="h-12 w-12 mb-4 text-red-400" />
                <p className="text-center text-sm">{error}</p>
                <Button 
                  onClick={startCamera} 
                  variant="outline" 
                  className="mt-4 bg-white/10 border-white/20 text-white hover:bg-white/20"
                >
                  Try Again
                </Button>
              </div>
            ) : !stream && !captured ? (
              <div className="absolute inset-0 flex items-center justify-center">
                {isLoading ? (
                  <div className="text-white">Starting camera...</div>
                ) : (
                  <div className="text-white">Camera access required</div>
                )}
              </div>
            ) : null}
            
            {stream && !captured && (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
            )}

            {captured && (
              <img
                src={captured}
                alt="Captured"
                className="w-full h-full object-cover"
              />
            )}
            
            <canvas ref={canvasRef} className="hidden" />
          </div>

          <div className="flex justify-center gap-3">
            {stream && !captured && !error && (
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