import { motion } from 'framer-motion'
import { TrendingUp, Brain, Activity } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

interface ConfidenceVisualizationProps {
  scores?: number[]
  prediction: string
  analysisType: 'face' | 'xray'
}

export function ConfidenceVisualization({ scores, prediction, analysisType }: ConfidenceVisualizationProps) {
  if (!scores || scores.length < 2) return null

  const confidence = Math.max(...scores) * 100
  const nonPcosScore = scores[0] * 100
  const pcosScore = scores[1] * 100

  const getConfidenceLevel = (conf: number) => {
    if (conf >= 90) return { level: 'Very High', color: 'from-emerald-500 to-teal-500', textColor: 'text-emerald-700' }
    if (conf >= 75) return { level: 'High', color: 'from-blue-500 to-indigo-500', textColor: 'text-blue-700' }
    if (conf >= 60) return { level: 'Moderate', color: 'from-amber-500 to-orange-500', textColor: 'text-amber-700' }
    return { level: 'Low', color: 'from-slate-400 to-gray-500', textColor: 'text-slate-600' }
  }

  const confidenceInfo = getConfidenceLevel(confidence)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="border-2 border-indigo-200 bg-gradient-to-br from-indigo-50 to-purple-50">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Brain className="h-5 w-5 text-indigo-600" />
            AI Confidence Analysis
            <Badge className={`ml-auto bg-gradient-to-r ${confidenceInfo.color} text-white`}>
              {confidence.toFixed(1)}% Confidence
            </Badge>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Overall Confidence Gauge */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="font-medium text-slate-700">Overall Confidence</span>
              <span className={`font-bold ${confidenceInfo.textColor}`}>
                {confidenceInfo.level}
              </span>
            </div>
            <div className="relative">
              <Progress 
                value={confidence} 
                className="h-4 bg-slate-200"
              />
              <div 
                className={`absolute top-0 left-0 h-4 rounded-full bg-gradient-to-r ${confidenceInfo.color} transition-all duration-1000 ease-out`}
                style={{ width: `${confidence}%` }}
              />
            </div>
          </div>

          {/* Detailed Score Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500"></div>
                <span className="text-sm font-medium">Healthy Indicators</span>
              </div>
              <div className="bg-white/70 p-3 rounded-lg">
                <div className="text-2xl font-bold text-emerald-600 mb-1">
                  {nonPcosScore.toFixed(1)}%
                </div>
                <Progress value={nonPcosScore} className="h-2 bg-slate-200" />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-orange-500 to-red-500"></div>
                <span className="text-sm font-medium">Risk Indicators</span>
              </div>
              <div className="bg-white/70 p-3 rounded-lg">
                <div className="text-2xl font-bold text-orange-600 mb-1">
                  {pcosScore.toFixed(1)}%
                </div>
                <Progress value={pcosScore} className="h-2 bg-slate-200" />
              </div>
            </div>
          </div>

          {/* Analysis Features */}
          <div className="bg-white/70 p-4 rounded-lg border border-indigo-200">
            <h4 className="font-semibold mb-3 flex items-center gap-2 text-slate-800">
              <Activity className="h-4 w-4 text-indigo-600" />
              Key Analysis Features
            </h4>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {analysisType === 'face' ? (
                <>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Facial symmetry</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Skin texture analysis</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Hormonal indicators</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Feature proportions</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Ovarian morphology</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Cyst detection</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Size measurements</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                    <span>Structural patterns</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}