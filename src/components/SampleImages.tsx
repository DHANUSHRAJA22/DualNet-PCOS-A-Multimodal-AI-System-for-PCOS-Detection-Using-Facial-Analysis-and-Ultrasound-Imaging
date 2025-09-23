import { useState } from 'react'
import { motion } from 'framer-motion'
import { TestTube, Download, Eye, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { toast } from 'sonner'
import { fixImageOrientation } from '@/lib/image'
import type { ProcessedImage } from '@/lib/image'

interface SampleImagesProps {
  onSelectFaceSample: (processedImage: ProcessedImage) => void
  onSelectXraySample: (processedImage: ProcessedImage) => void
}

const sampleImages = {
  face: [
    {
      id: 'face-sample-1',
      name: 'Sample Face 1',
      description: 'Clear frontal face photo - Normal indicators',
      url: 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=400',
      expectedResult: 'Low Risk'
    },
    {
      id: 'face-sample-2', 
      name: 'Sample Face 2',
      description: 'Professional headshot - Mixed indicators',
      url: 'https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg?auto=compress&cs=tinysrgb&w=400',
      expectedResult: 'Moderate Risk'
    }
  ],
  xray: [
    {
      id: 'xray-sample-1',
      name: 'Sample X-ray 1', 
      description: 'Pelvic X-ray - Normal morphology',
      url: 'https://images.pexels.com/photos/7089020/pexels-photo-7089020.jpeg?auto=compress&cs=tinysrgb&w=400',
      expectedResult: 'Low Risk'
    },
    {
      id: 'xray-sample-2',
      name: 'Sample X-ray 2',
      description: 'Medical imaging - Complex patterns',
      url: 'https://images.pexels.com/photos/7089021/pexels-photo-7089021.jpeg?auto=compress&cs=tinysrgb&w=400', 
      expectedResult: 'High Risk'
    }
  ]
}

export function SampleImages({ onSelectFaceSample, onSelectXraySample }: SampleImagesProps) {
  const [selectedPreview, setSelectedPreview] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState<string | null>(null)

  const handleSelectSample = async (type: 'face' | 'xray', sample: typeof sampleImages.face[0]) => {
    setIsLoading(sample.id)
    
    try {
      // Use image proxy to avoid CORS issues
      const proxyUrl = `/img-proxy?url=${encodeURIComponent(sample.url)}`
      const response = await fetch(proxyUrl)
      if (!response.ok) {
        throw new Error('Failed to fetch sample image')
      }
      
      const blob = await response.blob()
      
      // Create a File object
      const file = new File([blob], `${sample.name.toLowerCase().replace(/\s+/g, '-')}.jpg`, {
        type: 'image/jpeg',
        lastModified: Date.now(),
      })

      // Process the image properly using the same function as upload
      const processedImage = await fixImageOrientation(file)

      if (type === 'face') {
        onSelectFaceSample(processedImage)
      } else {
        onSelectXraySample(processedImage)
      }
      toast.success(`Sample ${type} image loaded successfully`)
    } catch (error) {
      toast.error(`Failed to load sample image: ${error instanceof Error ? error.message : 'Unknown error'}`)
      console.error('Sample image loading error:', error)
    } finally {
      setIsLoading(null)
    }
  }

  return (
    <Card className="border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-indigo-50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <TestTube className="h-5 w-5 text-purple-600" />
          Try Sample Images
          <Badge className="ml-auto bg-gradient-to-r from-purple-500 to-indigo-500 text-white">
            <Sparkles className="h-3 w-3 mr-1" />
            Demo Mode
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <p className="text-sm text-slate-600">
          Test the analyzer with pre-approved sample images. No personal data required.
        </p>

        {/* Face Samples */}
        <div className="space-y-3">
          <h4 className="font-semibold text-slate-800">Facial Analysis Samples</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {sampleImages.face.map((sample) => (
              <motion.div
                key={sample.id}
                whileHover={{ scale: 1.02 }}
                className="bg-white/70 p-3 rounded-lg border border-purple-200 hover:border-purple-300 transition-all"
              >
                <div className="flex items-center gap-3">
                  <img
                    src={sample.url}
                    alt={sample.name}
                    className="w-12 h-12 rounded-lg object-cover"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm text-slate-800">{sample.name}</div>
                    <div className="text-xs text-slate-600 truncate">{sample.description}</div>
                    <Badge variant="outline" className="text-xs mt-1">
                      Expected: {sample.expectedResult}
                    </Badge>
                  </div>
                  <div className="flex gap-1">
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                          <Eye className="h-3 w-3" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-md">
                        <DialogHeader>
                          <DialogTitle>{sample.name}</DialogTitle>
                        </DialogHeader>
                        <img
                          src={sample.url}
                          alt={sample.name}
                          className="w-full rounded-lg"
                        />
                        <p className="text-sm text-slate-600">{sample.description}</p>
                      </DialogContent>
                    </Dialog>
                    <Button
                      onClick={() => handleSelectSample('face', sample)}
                      disabled={isLoading === sample.id}
                      size="sm"
                      className="h-8 px-2 text-xs bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600"
                    >
                      {isLoading === sample.id ? (
                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" />
                      ) : (
                        <Download className="h-3 w-3" />
                      )}
                    </Button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* X-ray Samples */}
        <div className="space-y-3">
          <h4 className="font-semibold text-slate-800">X-ray Analysis Samples</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {sampleImages.xray.map((sample) => (
              <motion.div
                key={sample.id}
                whileHover={{ scale: 1.02 }}
                className="bg-white/70 p-3 rounded-lg border border-purple-200 hover:border-purple-300 transition-all"
              >
                <div className="flex items-center gap-3">
                  <img
                    src={sample.url}
                    alt={sample.name}
                    className="w-12 h-12 rounded-lg object-cover"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm text-slate-800">{sample.name}</div>
                    <div className="text-xs text-slate-600 truncate">{sample.description}</div>
                    <Badge variant="outline" className="text-xs mt-1">
                      Expected: {sample.expectedResult}
                    </Badge>
                  </div>
                  <div className="flex gap-1">
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                          <Eye className="h-3 w-3" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-md">
                        <DialogHeader>
                          <DialogTitle>{sample.name}</DialogTitle>
                        </DialogHeader>
                        <img
                          src={sample.url}
                          alt={sample.name}
                          className="w-full rounded-lg"
                        />
                        <p className="text-sm text-slate-600">{sample.description}</p>
                      </DialogContent>
                    </Dialog>
                    <Button
                      onClick={() => handleSelectSample('xray', sample)}
                      disabled={isLoading === sample.id}
                      size="sm"
                      className="h-8 px-2 text-xs bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600"
                    >
                      {isLoading === sample.id ? (
                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" />
                      ) : (
                        <Download className="h-3 w-3" />
                      )}
                    </Button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
          <p className="text-xs text-amber-800">
            <strong>Note:</strong> Sample images are for demonstration purposes only. 
            Results may not reflect actual medical conditions.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}