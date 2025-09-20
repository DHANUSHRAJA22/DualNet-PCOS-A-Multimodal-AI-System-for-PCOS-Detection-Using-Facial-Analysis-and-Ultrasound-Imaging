export interface ProcessedImage {
  file: File
  previewUrl: string
  width: number
  height: number
  size: number
  orientation: string
}

function getImageOrientation(file: File): Promise<number> {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const arrayBuffer = e.target?.result as ArrayBuffer
      const dataView = new DataView(arrayBuffer)
      
      // Check for JPEG signature
      if (dataView.getUint16(0) !== 0xFFD8) {
        resolve(1) // Default orientation
        return
      }
      
      let offset = 2
      let marker = dataView.getUint16(offset)
      
      while (marker !== 0xFFE1 && offset < dataView.byteLength) {
        offset += 2 + dataView.getUint16(offset + 2)
        if (offset >= dataView.byteLength) break
        marker = dataView.getUint16(offset)
      }
      
      if (marker !== 0xFFE1) {
        resolve(1)
        return
      }
      
      // Skip APP1 marker and length
      offset += 4
      
      // Check for Exif header
      if (dataView.getUint32(offset) !== 0x45786966) {
        resolve(1)
        return
      }
      
      offset += 6
      
      // Determine byte order
      const byteOrder = dataView.getUint16(offset)
      const littleEndian = byteOrder === 0x4949
      
      offset += 2
      
      // Skip TIFF header
      offset += 2
      
      // Get first IFD offset
      const ifdOffset = dataView.getUint32(offset, littleEndian)
      offset += ifdOffset - 8
      
      // Read number of directory entries
      const numEntries = dataView.getUint16(offset, littleEndian)
      offset += 2
      
      // Look for orientation tag (0x0112)
      for (let i = 0; i < numEntries; i++) {
        const tag = dataView.getUint16(offset, littleEndian)
        if (tag === 0x0112) {
          const orientation = dataView.getUint16(offset + 8, littleEndian)
          resolve(orientation)
          return
        }
        offset += 12
      }
      
      resolve(1) // Default orientation
    }
    
    reader.onerror = () => resolve(1)
    reader.readAsArrayBuffer(file.slice(0, 65536)) // Read first 64KB
  })
}

function orientationToText(orientation: number): string {
  const orientationMap: Record<number, string> = {
    1: 'Normal',
    2: 'Flip horizontal',
    3: 'Rotate 180°',
    4: 'Flip vertical',
    5: 'Rotate 90° CW + Flip horizontal',
    6: 'Rotate 90° CW',
    7: 'Rotate 90° CCW + Flip horizontal',
    8: 'Rotate 90° CCW',
  }
  
  return orientationMap[orientation] || 'Unknown'
}

function rotateCanvas(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, orientation: number): void {
  const { width, height } = canvas
  
  switch (orientation) {
    case 2:
      ctx.transform(-1, 0, 0, 1, width, 0)
      break
    case 3:
      ctx.transform(-1, 0, 0, -1, width, height)
      break
    case 4:
      ctx.transform(1, 0, 0, -1, 0, height)
      break
    case 5:
      canvas.width = height
      canvas.height = width
      ctx.transform(0, 1, 1, 0, 0, 0)
      break
    case 6:
      canvas.width = height
      canvas.height = width
      ctx.transform(0, 1, -1, 0, height, 0)
      break
    case 7:
      canvas.width = height
      canvas.height = width
      ctx.transform(0, -1, -1, 0, height, width)
      break
    case 8:
      canvas.width = height
      canvas.height = width
      ctx.transform(0, -1, 1, 0, 0, width)
      break
    default:
      break
  }
}

export async function fixImageOrientation(file: File): Promise<ProcessedImage> {
  try {
    const orientation = await getImageOrientation(file)
    
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        
        if (!ctx) {
          reject(new Error('Could not get canvas context'))
          return
        }
        
        // Set initial canvas dimensions
        canvas.width = img.width
        canvas.height = img.height
        
        // Apply rotation based on EXIF orientation
        if (orientation > 1) {
          rotateCanvas(canvas, ctx, orientation)
        }
        
        // Draw the image
        ctx.drawImage(img, 0, 0)
        
        // Convert canvas to blob
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error('Could not convert canvas to blob'))
            return
          }
          
          const newFile = new File([blob], file.name, {
            type: 'image/jpeg',
            lastModified: Date.now(),
          })
          
          const previewUrl = URL.createObjectURL(blob)
          
          resolve({
            file: newFile,
            previewUrl,
            width: canvas.width,
            height: canvas.height,
            size: blob.size,
            orientation: orientationToText(orientation),
          })
        }, 'image/jpeg', 0.92)
      }
      
      img.onerror = () => {
        reject(new Error('Could not load image'))
      }
      
      img.src = URL.createObjectURL(file)
    })
  } catch (error) {
    // Fallback: return original file with basic info
    const previewUrl = URL.createObjectURL(file)
    
    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        resolve({
          file,
          previewUrl,
          width: img.width,
          height: img.height,
          size: file.size,
          orientation: 'Unknown',
        })
      }
      img.onerror = () => {
        resolve({
          file,
          previewUrl,
          width: 0,
          height: 0,
          size: file.size,
          orientation: 'Unknown',
        })
      }
      img.src = previewUrl
    })
  }
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

export function isValidImageFile(file: File): boolean {
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
  return validTypes.includes(file.type)
}