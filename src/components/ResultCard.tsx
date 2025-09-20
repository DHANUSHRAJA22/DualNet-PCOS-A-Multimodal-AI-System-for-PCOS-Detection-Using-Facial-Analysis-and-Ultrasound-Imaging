import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { ConfidenceVisualization } from '@/components/ConfidenceVisualization'
import { withBase } from '@/lib/api'
import { cn } from '@/lib/utils'
import { motion } from 'framer-motion'

interface ResultCardProps {
  title: string
  prediction: string
  scores?: number[]
  originalImage?: string
  visualizationImage?: string
  foundLabels?: string[]
  riskLevel: 'low' | 'moderate' | 'high' | 'unknown'
  className?: string
}

function getRiskBadgeStyle(risk: string) {
  switch (risk) {
    case 'low':
      return 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600'
    case 'moderate':
      return 'bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-600 hover:to-orange-600'
    case 'high':
      return 'bg-gradient-to-r from-rose-500 to-red-500 text-white hover:from-rose-600 hover:to-red-600'
    default:
      return 'bg-gradient-to-r from-slate-400 to-gray-400 text-white hover:from-slate-500 hover:to-gray-500'
  }
}

function getRiskCardStyle(risk: string) {
  switch (risk) {
    case 'low':
      return 'border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50'
    case 'moderate':
      return 'border-amber-200 bg-gradient-to-br from-amber-50 to-orange-50'
    case 'high':
      return 'border-rose-200 bg-gradient-to-br from-rose-50 to-red-50'
    default:
      return 'border-slate-200 bg-gradient-to-br from-slate-50 to-gray-50'
  }
}

export function ResultCard({
  title,
  prediction,
  scores,
  originalImage,
  visualizationImage,
  foundLabels,
  riskLevel,
  className,
}: ResultCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className={cn(
        'h-fit hover:shadow-vibrant-lg transition-all duration-300 hover:-translate-y-1 border-2 shadow-lg',
        getRiskCardStyle(riskLevel),
        className
      )}>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-bold text-slate-800">{title}</CardTitle>
            <Badge 
              className={cn(
                'capitalize font-semibold px-4 py-2 shadow-lg',
                getRiskBadgeStyle(riskLevel)
              )}
            >
              {riskLevel === 'unknown' ? 'Pending' : `${riskLevel} Risk`}
            </Badge>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Prediction Text */}
          <div>
            <h4 className="font-semibold mb-3 text-slate-800">Analysis Result</h4>
            <p className="text-sm text-slate-700 bg-white/70 p-4 rounded-lg border border-slate-200 leading-relaxed">
              {prediction}
            </p>
          </div>

          {/* Enhanced Confidence Visualization */}
          <ConfidenceVisualization 
            scores={scores}
            prediction={prediction}
            analysisType={title.toLowerCase().includes('face') ? 'face' : 'xray'}
          />

          {/* Found Labels */}
          {foundLabels && foundLabels.length > 0 && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Detected Features</h4>
              <div className="flex flex-wrap gap-2">
                {foundLabels.map((label, index) => (
                  <Badge 
                    key={index} 
                    variant="outline" 
                    className="capitalize bg-white/70 border-slate-300 text-slate-700 hover:bg-slate-100 font-medium"
                  >
                    {label}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <Separator className="bg-slate-200" />

          {/* Images */}
          <div className="space-y-6">
            {originalImage && (
              <div>
                <h4 className="font-semibold mb-3 text-slate-800">Original Image</h4>
                <div className="relative group">
                  <img
                    src={withBase(originalImage)}
                    alt="Original"
                    className="w-full max-h-64 object-contain rounded-lg border-2 border-slate-200 bg-white shadow-lg transition-transform duration-300 group-hover:scale-105"
                    loading="lazy"
                  />
                </div>
              </div>
            )}

            {visualizationImage && (
              <div>
                <h4 className="font-semibold mb-3 text-slate-800">Analysis Visualization</h4>
                <div className="relative group">
                  <img
                    src={withBase(visualizationImage)}
                    alt="Analysis visualization"
                    className="w-full max-h-64 object-contain rounded-lg border-2 border-slate-200 bg-white shadow-lg transition-transform duration-300 group-hover:scale-105"
                    loading="lazy"
                  />
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}