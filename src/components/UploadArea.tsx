import { useState, useRef, useCallback } from 'react'
import { Upload, X, Camera, FileImage, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { CameraCapture } from '@/components/CameraCapture'
import { fixImageOrientation, formatFileSize, isValidImageFile, type ProcessedImage } from '@/lib/image'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import { motion } from 'framer-motion'

interface UploadAreaProps {
  id: string
  label: string
  subtext: string
  tips: string
  accept?: string
  onChange: (processedImage: ProcessedImage | null) => void
  className?: string
}

export function UploadArea({ 
  id, 
  label, 
  subtext, 
  tips, 
  accept = 'image/*', 
  onChange, 
  className 
}: UploadAreaProps) {
  const [processedImage, setProcessedImage] = useState<ProcessedImage | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const processFile = useCallback(async (file: File) => {
    if (!isValidImageFile(file)) {
      toast.error('Please select a valid image file (JPEG, PNG, or WebP)')
      return
    }

    // Validate file size (5MB limit)
    const maxSize = 5 * 1024 * 1024 // 5MB
    if (file.size > maxSize) {
      toast.error(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum allowed size (5MB)`)
      return
    }
    setIsProcessing(true)
    try {
      const processed = await fixImageOrientation(file)
      setProcessedImage(processed)
      onChange(processed)
      toast.success(`${label} image processed successfully`)
    } catch (error) {
      toast.error('Failed to process image')
      console.error('Image processing error:', error)
    } finally {
      setIsProcessing(false)
    }
  }, [onChange])

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      processFile(file)
    }
  }, [processFile])

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(false)
    
    const file = event.dataTransfer.files[0]
    if (file) {
      processFile(file)
    }
  }, [processFile])

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleCameraCapture = useCallback((file: File) => {
    processFile(file)
    setShowCamera(false)
  }, [processFile])

  const handleRemove = useCallback(() => {
    if (processedImage?.previewUrl) {
      URL.revokeObjectURL(processedImage.previewUrl)
    }
    setProcessedImage(null)
    onChange(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [processedImage, onChange])

  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  return (
    <div className={cn('space-y-4', className)}>
      <div className="space-y-2">
        <Label htmlFor={id} className="text-lg font-semibold">
          {label}
        </Label>
        <p className="text-sm text-muted-foreground">{subtext}</p>
        <div className="flex items-start gap-2 text-xs text-muted-foreground">
          <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
          <span>{tips}</span>
        </div>
      </div>
      
      <Card
        className={cn(
          'border-2 border-dashed transition-all duration-200 cursor-pointer hover:border-primary/50 hover:shadow-md',
          isDragOver && 'border-primary bg-primary/5 shadow-lg',
          processedImage && 'border-solid border-border'
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <CardContent className="p-6">
          {!processedImage ? (
            <div className="text-center space-y-4">
              {isProcessing ? (
                <motion.div 
                  className="space-y-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <div className="flex justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                  </div>
                  <p className="text-sm text-muted-foreground">Processing image...</p>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  <div className="flex justify-center">
                    <div className="p-4 rounded-full bg-primary/10">
                      <Upload className="h-8 w-8 text-primary" />
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium mb-2">
                      Click to upload or drag & drop
                    </p>
                    <p className="text-xs text-muted-foreground mb-4">
                      Supports JPEG, PNG, WebP (max 5MB)
                    </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          ) : (
            <motion.div 
              className="space-y-4"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <div className="relative">
                <img
                  src={processedImage.previewUrl}
                  alt="Preview"
                  className="w-full max-h-48 object-contain rounded-lg bg-muted"
                />
                <Button
                  onClick={handleRemove}
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2 h-8 w-8 shadow-lg"
                  aria-label="Remove image"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              
              <div className="text-sm space-y-1 bg-muted/50 p-3 rounded-md">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Dimensions:</span>
                  <span>{processedImage.width} × {processedImage.height}px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Size:</span>
                  <span>{formatFileSize(processedImage.size)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Orientation:</span>
                  <span>{processedImage.orientation}</span>
                </div>
              </div>
            </motion.div>
          )}
        </CardContent>
      </Card>

      <input
        ref={fileInputRef}
        id={id}
        type="file"
        accept={accept}
        onChange={handleFileSelect}
        className="sr-only"
        aria-describedby={`${id}-description`}
      />

      <CameraCapture
        open={showCamera}
        onOpenChange={setShowCamera}
        onCapture={handleCameraCapture}
      />
    </div>
  )
}