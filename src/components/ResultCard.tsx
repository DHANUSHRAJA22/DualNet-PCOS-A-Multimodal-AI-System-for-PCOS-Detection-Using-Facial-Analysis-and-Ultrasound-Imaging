import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { ConfidenceVisualization } from '@/components/ConfidenceVisualization'
import { withBase } from '@/lib/api'
import type { ModalityResult } from '@/lib/api'
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
  confidence?: number // Overall confidence from final result
  thresholds?: { low: number; high: number } // Backend risk thresholds
  modality?: ModalityResult // Full modality data for enhanced display
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
  confidence,
  thresholds = { low: 0.33, high: 0.66 },
  modality,
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

          {/* Gender Information for Face Analysis */}
          {modality?.gender && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Gender Detection</h4>
              <div className="bg-white/70 p-3 rounded-lg border border-slate-200">
                <div className="flex justify-between items-center">
                  <span className="text-sm">Detected Gender:</span>
                  <Badge variant="outline" className="capitalize">
                    {modality.gender.label} ({(Math.max(modality.gender.male, modality.gender.female) * 100).toFixed(1)}%)
                  </Badge>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced Confidence Visualization */}
          <ConfidenceVisualization 
            scores={scores}
            prediction={prediction}
            analysisType={title.toLowerCase().includes('face') ? 'face' : 'xray'}
            confidence={modality?.ensemble?.score || confidence}
            ensemble={modality?.ensemble}
            thresholds={thresholds}
          />

          {/* Per-Model Scores Display */}
          {(modality?.per_model || modality?.face_models || modality?.xray_models) && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Individual Model Scores</h4>
              <div className="bg-white/70 rounded-lg p-4 border border-slate-200">
                <div className="space-y-3">
                  {Object.entries(modality.per_model || modality.face_models || modality.xray_models || {}).map(([modelName, score]) => {
                    // Handle both score formats (single number or array)
                    const displayScore = Array.isArray(score) ? score[1] : score; // Use PCOS probability
                    return (
                    <div key={modelName} className="flex justify-between items-center">
                      <span className="text-sm font-medium capitalize text-slate-700">
                        {modelName.replace('_', ' ')}
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-slate-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
                            style={{ width: `${displayScore * 100}%` }}
                          />
                        </div>
                        <Badge variant="outline" className="font-mono text-xs">
                          {(displayScore * 100).toFixed(1)}%
                        </Badge>
                      </div>
                    </div>
                  )})}
                  
                  {modality.ensemble && (
                    <div className="pt-3 border-t border-slate-200">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-semibold text-slate-800">Ensemble Result</span>
                        <Badge className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white">
                          {(modality.ensemble.score * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <div className="text-xs text-slate-600 mt-1">
                        Method: {modality.ensemble.method} • Models: {modality.ensemble.models_used}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

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

          {/* YOLO Detections for X-ray */}
          {modality?.detections && modality.detections.length > 0 && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Object Detections</h4>
              <div className="space-y-2">
                {modality.detections.map((detection, index) => (
                  <div key={index} className="bg-white/70 p-3 rounded-lg border border-slate-200">
                    <div className="flex justify-between items-center">
                      <span className="capitalize font-medium">{detection.label}</span>
                      <Badge variant="outline">
                        {(detection.conf * 100).toFixed(1)}% confidence
                      </Badge>
                    </div>
                    <div className="text-xs text-slate-600 mt-1">
                      Box: [{detection.box.map(n => Math.round(n)).join(', ')}]
                    </div>
                  </div>
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