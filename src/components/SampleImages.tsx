// src/components/SampleImages.tsx
import { useRef, useState } from 'react'
import type { ProcessedImage } from '@/lib/image'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import { ImageIcon, UploadCloud } from 'lucide-react'

type Props = {
  onSelectFaceSample?: (img: ProcessedImage) => void
  onSelectXraySample?: (img: ProcessedImage) => void
}

type Sample = { url: string; title: string; subtitle?: string }

// === Your actual files (from /public/samples/...) ===
// Face
const FACE_SAMPLES: Sample[] = [
  { url: '/samples/face/44.jpg', title: 'Face sample', subtitle: 'neutral, front-facing' },
  { url: '/samples/face/33.jpeg', title: 'Face sample', subtitle: 'good lighting' },
  { url: '/samples/face/pimple-1658939057.jpg', title: 'Face sample', subtitle: 'no glasses' },
]

// X-ray
const XRAY_SAMPLES: Sample[] = [
  { url: '/samples/xray/img_0_182.jpg', title: 'X-ray sample', subtitle: 'pelvic AP view' },
  { url: '/samples/xray/img_0_60.jpg', title: 'X-ray sample', subtitle: 'good contrast' },
  // file name has spaces → URL-encoded when fetching
  { url: '/samples/xray/Copy of img_0_9989.jpg', title: 'X-ray sample', subtitle: 'centered field' },
]

// ---- helpers ----
function toProcessedImage(file: File): ProcessedImage {
  const previewUrl = URL.createObjectURL(file)
  return {
    file,
    previewUrl,
    size: file.size,
    mime: file.type || 'image/jpeg',
  }
}

async function urlToProcessedImage(url: string): Promise<ProcessedImage> {
  // encode only for fetching (images display fine with spaces in src)
  const res = await fetch(encodeURI(url), { cache: 'no-store' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const blob = await res.blob()
  const name = url.split('/').pop() || 'sample.jpg'
  const file = new File([blob], name, { type: blob.type || 'image/jpeg' })
  return toProcessedImage(file)
}

// ---- component ----
export function SampleImages({ onSelectFaceSample, onSelectXraySample }: Props) {
  const [loadingKey, setLoadingKey] = useState<string | null>(null)
  const faceInputRef = useRef<HTMLInputElement | null>(null)
  const xrayInputRef = useRef<HTMLInputElement | null>(null)

  const pickUrl = async (kind: 'face' | 'xray', url: string) => {
    try {
      setLoadingKey(`${kind}:${url}`)
      const img = await urlToProcessedImage(url)
      if (kind === 'face') {
        onSelectFaceSample?.(img)
        toast.success('Face sample selected')
      } else {
        onSelectXraySample?.(img)
        toast.success('X-ray sample selected')
      }
    } catch (e: any) {
      toast.error(`Failed to load sample: ${e?.message || e}`)
    } finally {
      setLoadingKey(null)
    }
  }

  const pickLocal = (kind: 'face' | 'xray', files: FileList | null) => {
    if (!files?.length) return
    const file = files[0]
    const max = 5 * 1024 * 1024
    if (file.size > max) {
      toast.error(`File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Max 5MB.`)
      return
    }
    const img = toProcessedImage(file)
    if (kind === 'face') {
      onSelectFaceSample?.(img)
      toast.success('Face image imported')
    } else {
      onSelectXraySample?.(img)
      toast.success('X-ray image imported')
    }
  }

  const Tile = ({ s, kind }: { s: Sample; kind: 'face' | 'xray' }) => (
    <button
      onClick={() => pickUrl(kind, s.url)}
      className="relative group rounded-lg overflow-hidden border border-dashed border-slate-200 bg-slate-50/60 hover:bg-slate-50 hover:shadow-md transition"
      title="Click to use this sample"
    >
      <img
        src={s.url}
        alt={s.title}
        className="aspect-square object-cover w-full group-hover:scale-105 transition-transform duration-200"
        loading="lazy"
      />
      <div className="absolute inset-0 flex flex-col items-center justify-center text-[11px] text-slate-500 select-none pointer-events-none">
        <div className="font-medium text-slate-700">{s.title}</div>
        {s.subtitle && <div className="opacity-70">{s.subtitle}</div>}
      </div>
      {loadingKey === `${kind}:${s.url}` && (
        <div className="absolute inset-0 grid place-items-center bg-white/70 text-sm">Loading…</div>
      )}
    </button>
  )

  return (
    <div className="space-y-8">
      <Card className="border-0 shadow-vibrant-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ImageIcon className="w-5 h-5" />
            Sample Images
            <Badge variant="secondary" className="ml-2">Click to use</Badge>
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-10">
          {/* Face */}
          <section>
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-medium text-slate-700">Face</div>
              <div className="flex items-center gap-2">
                <input
                  ref={faceInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => pickLocal('face', e.target.files)}
                />
                <Button variant="outline" size="sm" onClick={() => faceInputRef.current?.click()}>
                  <UploadCloud className="w-4 h-4 mr-1" />
                  Add from PC
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {FACE_SAMPLES.map((s) => (
                <Tile key={s.url} s={s} kind="face" />
              ))}
            </div>
          </section>

          {/* X-ray */}
          <section>
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-medium text-slate-700">X-ray</div>
              <div className="flex items-center gap-2">
                <input
                  ref={xrayInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => pickLocal('xray', e.target.files)}
                />
                <Button variant="outline" size="sm" onClick={() => xrayInputRef.current?.click()}>
                  <UploadCloud className="w-4 h-4 mr-1" />
                  Add from PC
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {XRAY_SAMPLES.map((s) => (
                <Tile key={s.url} s={s} kind="xray" />
              ))}
            </div>
          </section>

          <div className="text-right">
            <Button
              variant="outline"
              onClick={() => window.open('/samples/', '_blank', 'noopener,noreferrer')}
            >
              Open samples folder
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
