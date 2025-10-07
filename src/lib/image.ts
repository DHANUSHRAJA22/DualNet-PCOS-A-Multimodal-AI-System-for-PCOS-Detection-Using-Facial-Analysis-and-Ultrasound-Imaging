// src/lib/image.ts
export type ProcessedImage = {
  file: File
  previewUrl: string
  size: number
  mime: string
  width?: number
  height?: number
}

/** Fetch an image URL into a File object (keeps filename if possible). */
export async function fileFromUrl(url: string, filename?: string, mime?: string): Promise<File> {
  const res = await fetch(url, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Failed to fetch sample image: ${res.status}`)
  const blob = await res.blob()
  const type = mime || blob.type || 'image/jpeg'
  const name =
    filename ||
    url.split('/').pop()?.split('?')[0] ||
    `sample-${Math.random().toString(36).slice(2)}.jpg`
  return new File([blob], name, { type })
}

/** Convert a File to our ProcessedImage shape (adds preview + dimensions). */
export async function toProcessedImage(file: File): Promise<ProcessedImage> {
  const previewUrl = URL.createObjectURL(file)
  const dims = await new Promise<{ w: number; h: number }>((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ w: img.width, h: img.height })
    img.src = previewUrl
  })
  return {
    file,
    previewUrl,
    size: file.size,
    mime: file.type,
    width: dims.w,
    height: dims.h,
  }
}

/** Convenience: go from URL straight to ProcessedImage. */
export async function processedFromUrl(url: string, filename?: string): Promise<ProcessedImage> {
  const file = await fileFromUrl(url, filename)
  return toProcessedImage(file)
}

/** Clean up a ProcessedImage’s object URL when you’re done with it. */
export function revokeProcessedImage(p?: ProcessedImage | null) {
  if (p?.previewUrl) URL.revokeObjectURL(p.previewUrl)
}
