import { useLocation, Navigate, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Brain, ScanLine, CheckCircle, AlertTriangle, TrendingUp, Eye, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { ResultCard } from '@/components/ResultCard'
import { MedicalDisclaimer } from '@/components/MedicalDisclaimer'
import { RiskGauge } from '@/components/RiskGauge'
import { AIPerformanceMetrics } from '@/components/AIPerformanceMetrics'
import { withBase } from '@/lib/api'
import type { StructuredPredictionResponse, LegacyPredictionResponse, ModalityResult, ROIResult } from '@/lib/api'

type RiskLevel = 'low' | 'moderate' | 'high' | 'unknown'

function isStructuredResponse(response: any): response is StructuredPredictionResponse {
  return response && 'modalities' in response && 'final' in response
}

function getRiskLevel(combinedResult?: string): RiskLevel {
  if (!combinedResult) return 'unknown'
  
  const lowRiskKeywords = ['low risk', 'no pcos', 'non-pcos', 'negative', 'no symptoms']
  const highRiskKeywords = ['high risk', 'positive', 'pcos detected', 'likely pcos']
  const moderateRiskKeywords = ['moderate', 'possible', 'may suggest']
  
  const lower = combinedResult.toLowerCase()
  
  if (lowRiskKeywords.some(keyword => lower.includes(keyword))) {
    return 'low'
  }
  
  if (highRiskKeywords.some(keyword => lower.includes(keyword))) {
    return 'high'
  }
  
  if (moderateRiskKeywords.some(keyword => lower.includes(keyword))) {
    return 'moderate'
  }
  
  return 'unknown'
}

function getRiskExplanation(riskLevel: RiskLevel): string {
  switch (riskLevel) {
    case 'low':
      return 'The analysis shows minimal indicators typically associated with PCOS. Regular monitoring and healthy lifestyle practices are recommended. Continue with routine healthcare checkups.'
    case 'moderate':
      return 'The analysis shows some potential indicators that may warrant further investigation. Consider consulting with a healthcare professional for comprehensive evaluation and additional testing.'
    case 'high':
      return 'The analysis indicates several markers that could be associated with PCOS symptoms. Professional medical consultation is strongly recommended for proper diagnosis and treatment planning.'
    default:
      return 'The analysis results are inconclusive. Additional testing or consultation with healthcare professionals may be needed for proper evaluation.'
  }
}

function getAnalysisSummary(prediction?: string): { status: 'normal' | 'review'; text: string } {
  if (!prediction) return { status: 'review', text: 'No data' }
  
  const lower = prediction.toLowerCase()
  const normalKeywords = ['normal', 'healthy', 'no symptoms', 'negative', 'non-pcos']
  
  if (normalKeywords.some(keyword => lower.includes(keyword))) {
    return { status: 'normal', text: 'Normal' }
  }
  
  return { status: 'review', text: 'Review Needed' }
}

function PerModelBreakdown({ modality }: { modality: ModalityResult }) {
  if (!modality.per_model || Object.keys(modality.per_model).length === 0) {
    return null
  }

  return (
    <Card className="border-indigo-200 bg-gradient-to-br from-indigo-50 to-purple-50">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <BarChart3 className="h-5 w-5 text-indigo-600" />
          Per-Model Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {Object.entries(modality.per_model).map(([modelName, score]) => (
          <div key={modelName} className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="font-medium capitalize text-slate-700">
                {modelName.replace('_', ' ')}
              </span>
              <Badge variant="outline" className="font-mono">
                {(score * 100).toFixed(1)}%
              </Badge>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
                style={{ width: `${score * 100}%` }}
              />
            </div>
          </div>
        ))}
        
        {modality.ensemble && (
          <div className="mt-4 pt-4 border-t border-indigo-200">
            <div className="flex justify-between items-center mb-2">
              <span className="font-semibold text-slate-800">Ensemble Result</span>
              <Badge className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white">
                {(modality.ensemble.score * 100).toFixed(1)}%
              </Badge>
            </div>
            <div className="text-sm text-slate-600">
              <div>Method: {modality.ensemble.method}</div>
              <div>Models used: {modality.ensemble.models_used}</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ROIBreakdown({ rois }: { rois: ROIResult[] }) {
  if (!rois || rois.length === 0) {
    return null
  }

  return (
    <Card className="border-teal-200 bg-gradient-to-br from-teal-50 to-emerald-50">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Eye className="h-5 w-5 text-teal-600" />
          Region of Interest Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Accordion type="single" collapsible className="space-y-2">
          {rois.map((roi) => (
            <AccordionItem key={roi.roi_id} value={`roi-${roi.roi_id}`} className="border border-teal-200 rounded-lg">
              <AccordionTrigger className="px-4 py-3 hover:no-underline">
                <div className="flex items-center justify-between w-full mr-4">
                  <span className="font-medium">ROI #{roi.roi_id + 1}</span>
                  <Badge className="bg-gradient-to-r from-teal-500 to-emerald-500 text-white">
                    {(roi.ensemble.score * 100).toFixed(1)}% Risk
                  </Badge>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-4 pb-4">
                <div className="space-y-3">
                  <div className="text-sm text-slate-600">
                    <strong>Bounding Box:</strong> [{roi.box.map(n => n.toFixed(0)).join(', ')}]
                  </div>
                  
                  <div className="space-y-2">
                    <div className="font-medium text-slate-700">Model Predictions:</div>
                    {Object.entries(roi.per_model).map(([modelName, score]) => (
                      <div key={modelName} className="flex justify-between items-center">
                        <span className="text-sm capitalize">{modelName.replace('_', ' ')}</span>
                        <span className="font-mono text-sm">{(score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                  
                  <div className="pt-2 border-t border-teal-200">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Ensemble ({roi.ensemble.method})</span>
                      <Badge variant="outline">{(roi.ensemble.score * 100).toFixed(1)}%</Badge>
                    </div>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </CardContent>
    </Card>
  )
}

const Reveal = ({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
  >
    {children}
  </motion.div>
)

export function ResultsPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const results = location.state?.results as StructuredPredictionResponse | LegacyPredictionResponse

  if (!results) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex items-center justify-center">
        <Card className="max-w-md mx-4 card-gradient shadow-vibrant-lg">
          <CardHeader className="text-center">
            <CardTitle className="text-gradient">No Results Found</CardTitle>
            <CardDescription>
              Please upload images first to see analysis results.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center">
            <Button 
              onClick={() => navigate('/')}
              className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-vibrant"
            >
              Return to Upload
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Handle both structured and legacy responses
  let overallRisk: RiskLevel
  let overallConfidence: number
  let explanation: string
  let modalities: ModalityResult[] = []
  let processingTime: number = 0
  let debugInfo: any = {}

  if (isStructuredResponse(results)) {
    // New structured format
    overallRisk = results.final.overall_risk as RiskLevel
    overallConfidence = results.final.confidence
    explanation = results.final.explanation
    modalities = results.modalities
    processingTime = results.processing_time_ms
    debugInfo = results.debug
  } else {
    // Legacy format
    overallRisk = (results.overall_risk as RiskLevel) || getRiskLevel(results.combined)
    // Calculate confidence from available scores
    let maxScore = 0.5
    if (results.face_scores && results.face_scores.length > 0) {
      maxScore = Math.max(maxScore, Math.max(...results.face_scores))
    }
    overallConfidence = maxScore
    explanation = results.combined || 'Analysis completed'
    
    // Convert legacy to modality format for display
    if (results.face_pred) {
      modalities.push({
        type: 'face',
        label: results.face_pred,
        scores: results.face_scores || [],
        risk: results.face_risk as RiskLevel || 'unknown',
        original_img: results.face_img
      })
    }
    
    if (results.xray_pred) {
      modalities.push({
        type: 'xray',
        label: results.xray_pred,
        scores: [],
        risk: results.xray_risk as RiskLevel || 'unknown',
        original_img: results.xray_img,
        visualization: results.yolo_vis,
        found_labels: results.found_labels
      })
    }
  }

  const faceSummary = getAnalysisSummary(modalities.find(m => m.type === 'face')?.label)
  const xraySummary = getAnalysisSummary(modalities.find(m => m.type === 'xray')?.label)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50">
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-white/95 backdrop-blur-sm border-b border-purple-100">
        <div className="container mx-auto px-4 py-4">
          <Button 
            variant="ghost" 
            onClick={() => navigate('/')}
            className="font-medium hover:bg-purple-50 text-slate-700 hover:text-purple-700"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Upload
          </Button>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Overall Assessment */}
          <Reveal>
            <Card className="border-2 border-orange-200 bg-gradient-to-r from-orange-50 to-amber-50 shadow-vibrant-lg">
              <CardHeader>
                <CardTitle className="text-2xl flex items-center gap-3 text-slate-800">
                  <div className="p-3 rounded-lg bg-gradient-to-br from-orange-400 to-amber-400 text-white shadow-lg">
                    <ScanLine className="h-6 w-6" />
                  </div>
                  Overall Assessment
                  {processingTime > 0 && (
                    <Badge variant="outline" className="ml-auto text-xs">
                      {processingTime.toFixed(0)}ms
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              
              <CardContent className="space-y-6">
                <div className="flex items-start gap-4">
                  <Badge 
                    className={`text-base px-6 py-3 capitalize font-semibold shadow-lg ${
                      overallRisk === 'low' 
                        ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600' 
                        : overallRisk === 'high' 
                        ? 'bg-gradient-to-r from-rose-500 to-red-500 text-white hover:from-rose-600 hover:to-red-600' 
                        : 'bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-600 hover:to-orange-600'
                    }`}
                  >
                    <TrendingUp className="h-4 w-4 mr-2" />
                    {overallRisk === 'unknown' ? 'Pending Analysis' : `${overallRisk} Risk`}
                  </Badge>
                </div>
                
                <div className="bg-white/70 rounded-lg p-6 border border-orange-200">
                  <p className="text-lg leading-relaxed mb-4 text-slate-700">
                    {explanation}
                  </p>
                </div>
                
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
                  <h4 className="font-semibold mb-3 text-lg text-slate-800">Detailed Explanation</h4>
                  <p className="text-slate-600 leading-relaxed">
                    {getRiskExplanation(overallRisk)}
                  </p>
                </div>
                
                {/* Analysis Summary Chips */}
                <div className="flex flex-wrap gap-4">
                  {modalities.find(m => m.type === 'face') && (
                    <motion.div 
                      className={`flex items-center gap-3 px-4 py-3 rounded-full shadow-lg border-2 ${
                        faceSummary.status === 'normal' 
                          ? 'bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200' 
                          : 'bg-gradient-to-r from-orange-50 to-amber-50 border-orange-200'
                      }`}
                      whileHover={{ scale: 1.02 }}
                    >
                      {faceSummary.status === 'normal' ? (
                        <CheckCircle className="h-5 w-5 text-emerald-600" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-orange-600" />
                      )}
                      <span className="text-sm font-medium text-slate-700">
                        <span className="font-semibold">Facial Analysis:</span> {faceSummary.text}
                      </span>
                    </motion.div>
                  )}
                  
                  {modalities.find(m => m.type === 'xray') && (
                    <motion.div 
                      className={`flex items-center gap-3 px-4 py-3 rounded-full shadow-lg border-2 ${
                        xraySummary.status === 'normal' 
                          ? 'bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200' 
                          : 'bg-gradient-to-r from-orange-50 to-amber-50 border-orange-200'
                      }`}
                      whileHover={{ scale: 1.02 }}
                    >
                      {xraySummary.status === 'normal' ? (
                        <CheckCircle className="h-5 w-5 text-emerald-600" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-orange-600" />
                      )}
                      <span className="text-sm font-medium text-slate-700">
                        <span className="font-semibold">Imaging Analysis:</span> {xraySummary.text}
                      </span>
                    </motion.div>
                  )}
                </div>
              </CardContent>
            </Card>
          </Reveal>

          {/* Risk Gauge Visualization */}
          <Reveal delay={0.05}>
            <div className="flex justify-center">
              <RiskGauge 
                riskLevel={overallRisk} 
                confidence={overallConfidence}
                className="max-w-sm"
              />
            </div>
          </Reveal>

          {/* Detailed Results */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {modalities.map((modality, index) => (
              <Reveal delay={0.1}>
                <ResultCard
                  key={modality.type}
                  title={modality.type === 'face' ? 'Facial Analysis' : 'X-ray Analysis'}
                  prediction={modality.label}
                  scores={modality.scores}
                  originalImage={modality.original_img}
                  visualizationImage={modality.visualization}
                  foundLabels={modality.found_labels}
                  riskLevel={modality.risk as RiskLevel}
                  confidence={overallConfidence}
                  modality={modality}
                />
              </Reveal>
            ))}
          </div>

          {/* Per-Model and ROI Breakdowns */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {modalities.map((modality, index) => (
              <Reveal key={modality.type} delay={0.3 + index * 0.1}>
                <div className="space-y-4">
                  <PerModelBreakdown modality={modality} />
                  {modality.type === 'xray' && modality.per_roi && (
                    <ROIBreakdown rois={modality.per_roi} />
                  )}
                </div>
              </Reveal>
            ))}
          </div>

          {/* AI Performance Metrics */}
          <Reveal delay={0.4}>
            <AIPerformanceMetrics />
          </Reveal>

          {/* Actions */}
          <Reveal delay={0.5}>
            <div className="text-center">
              <Button 
                onClick={() => navigate('/')}
                size="lg"
                className="px-12 py-6 text-lg bg-gradient-to-r from-purple-600 via-indigo-600 to-teal-600 hover:from-purple-700 hover:via-indigo-700 hover:to-teal-700 shadow-vibrant-lg transition-all duration-300 hover:scale-105"
              >
                <Brain className="h-5 w-5 mr-2" />
                Analyze Another Image
              </Button>
            </div>
          </Reveal>

          {/* Medical Disclaimer */}
          <Reveal delay={0.6}>
            <MedicalDisclaimer />
          </Reveal>

          {/* Debug Information (Development Only) */}
          {debugInfo && Object.keys(debugInfo).length > 0 && (
            <Reveal delay={0.7}>
              <Card className="border-slate-200 bg-slate-50">
                <CardHeader>
                  <CardTitle className="text-lg text-slate-700">Debug Information</CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="text-xs text-slate-600 overflow-auto">
                    {JSON.stringify(debugInfo, null, 2)}
                  </pre>
                </CardContent>
              </Card>
            </Reveal>
          )}
        </div>
      </div>
    </div>
  )
}