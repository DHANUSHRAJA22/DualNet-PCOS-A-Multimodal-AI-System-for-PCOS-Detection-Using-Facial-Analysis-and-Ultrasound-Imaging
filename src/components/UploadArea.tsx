import React, { useCallback, useEffect, useRef, useState } from 'react'
import { UploadCloud, Camera, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { toast } from 'sonner'
import type { ProcessedImage } from '@/lib/image'

type Props = {
  id: string
  label: string
  subtext?: string
  tips?: string
  onChange: (img: ProcessedImage | null) => void
  value?: ProcessedImage | null
  onOpenCamera?: () => void
  accept?: string
  maxMb?: number
}

export function UploadArea({
  id,
  label,
  subtext,
  tips,
  onChange,
  value,
  onOpenCamera,
  accept = 'image/*',
  maxMb = 5,
}: Props) {
  const [localPreview, setLocalPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    setLocalPreview(value?.previewUrl || null)
  }, [value])

  const revokeLocal = () => {
    if (localPreview && (!value || value.previewUrl !== localPreview)) {
      URL.revokeObjectURL(localPreview)
    }
  }

  useEffect(() => {
    return () => revokeLocal()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return
      const file = files[0]
      const sizeMb = file.size / 1024 / 1024
      if (sizeMb > maxMb) {
        toast.error(`Maximum size is ${maxMb}MB`)
        return
      }
      const previewUrl = URL.createObjectURL(file)
      setLocalPreview(previewUrl)
      onChange({
        file,
        previewUrl,
        size: file.size,
        mime: file.type,
      })
    },
    [maxMb, onChange],
  )

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  const onPick = () => inputRef.current?.click()

  const clear = () => {
    revokeLocal()
    setLocalPreview(null)
    onChange(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <Card className="p-4 border-dashed border-2">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="font-semibold text-slate-800">{label}</div>
          {subtext && <div className="text-sm text-slate-500">{subtext}</div>}
        </div>
        <div className="flex gap-2">
          {onOpenCamera && (
            <Button variant="outline" size="sm" onClick={onOpenCamera}>
              <Camera className="w-4 h-4 mr-1" />
              Camera
            </Button>
          )}
          {localPreview && (
            <Button
              variant="ghost"
              size="icon"
              onClick={clear}
              title="Clear image"
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          'rounded-lg overflow-hidden bg-muted/40 border',
          dragging ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200',
          'grid place-items-center aspect-square',
        ].join(' ')}
      >
        {localPreview ? (
          <img
            src={localPreview}
            alt="Selected"
            className="object-contain w-full h-full bg-white"
            loading="lazy"
          />
        ) : (
          <button
            type="button"
            onClick={onPick}
            className="flex flex-col items-center justify-center text-slate-600 hover:text-slate-800 focus:outline-none p-6"
          >
            <UploadCloud className="w-10 h-10 mb-2 opacity-70" />
            <div className="font-medium">Drop image here or click to browse</div>
            {tips && <div className="text-xs mt-1 text-slate-500">{tips}</div>}
          </button>
        )}
      </div>

      <input
        id={`${id}-input`}
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </Card>
  )
}
