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
  thresholds?: { low: number; high: number }
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

// clamp to [0,1] and format to percent text
const asPct = (x?: number) =>
  typeof x === 'number' && isFinite(x) ? `${(Math.max(0, Math.min(1, x)) * 100).toFixed(1)}%` : '0.0%'

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
  // Access optional backend fields without TypeScript complaints
  const m: any = modality || {}

  // Prefer fused backend score for X-ray: ensemble_score → ensemble.score → fallback to prop confidence
  const fusedScore: number | undefined =
    typeof m.ensemble_score === 'number'
      ? m.ensemble_score
      : typeof m.ensemble?.score === 'number'
      ? m.ensemble.score
      : typeof confidence === 'number'
      ? confidence
      : undefined

  // Per-model scores (object → array), normalize to [0..1]
  const perModel: { name: string; score: number }[] = Object.entries(m.per_model ?? {}).map(
    ([name, score]: [string, any]) => {
      const raw = Array.isArray(score) ? Number(score[1]) : Number(score)
      const val = isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0
      return { name, score: val }
    }
  )

  // Group YOLO detections by label for a compact summary (e.g., "Cyst ×7")
  const detections: Array<{ label?: string; conf?: number; box?: number[] }> = m.detections ?? []
  const groupedDetections = (() => {
    if (!detections?.length) return [] as Array<{ label: string; count: number }>
    const map = new Map<string, number>()
    for (const d of detections) {
      const key = String(d?.label ?? 'object').toLowerCase()
      map.set(key, (map.get(key) ?? 0) + 1)
    }
    return [...map.entries()]
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count)
  })()

  // Prefer annotated YOLO image; fall back to the original
  const origImg = originalImage ?? m.xray_img
  const visImg = visualizationImage ?? m.yolo_vis

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <Card
        className={cn(
          'h-fit hover:shadow-vibrant-lg transition-all duration-300 hover:-translate-y-1 border-2 shadow-lg',
          getRiskCardStyle(riskLevel),
          className
        )}
      >
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-bold text-slate-800">{title}</CardTitle>
            <Badge className={cn('capitalize font-semibold px-4 py-2 shadow-lg', getRiskBadgeStyle(riskLevel))}>
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

          {/* Gender (face analysis only) */}
          {m?.gender && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Gender Detection</h4>
              <div className="bg-white/70 p-3 rounded-lg border border-slate-200">
                <div className="flex justify-between items-center">
                  <span className="text-sm">Detected Gender:</span>
                  <Badge variant="outline" className="capitalize">
                    {m.gender.label} ({(Math.max(m.gender.male, m.gender.female) * 100).toFixed(1)}%)
                  </Badge>
                </div>
              </div>
            </div>
          )}

          {/* Confidence / Ensemble viz */}
          <ConfidenceVisualization
            scores={scores}
            prediction={prediction}
            analysisType={title.toLowerCase().includes('face') ? 'face' : 'xray'}
            confidence={typeof fusedScore === 'number' ? fusedScore : 0}
            ensemble={m?.ensemble}
            thresholds={thresholds}
          />

          {/* Per-Model Scores */}
          {perModel.length > 0 && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Individual Model Scores</h4>
              <div className="bg-white/70 rounded-lg p-4 border border-slate-200">
                <div className="space-y-3">
                  {perModel.map(({ name, score }) => (
                    <div key={name} className="flex justify-between items-center">
                      <span className="text-sm font-medium capitalize text-slate-700">
                        {name.replace(/_/g, ' ')}
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-slate-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
                            style={{ width: `${Math.round(score * 100)}%` }}
                          />
                        </div>
                        <Badge variant="outline" className="font-mono text-xs">
                          {asPct(score)}
                        </Badge>
                      </div>
                    </div>
                  ))}

                  {/* Ensemble row (uses fused score when available) */}
                  {(typeof fusedScore === 'number' || m?.ensemble) && (
                    <div className="pt-3 border-t border-slate-200">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-semibold text-slate-800">Ensemble Result</span>
                        <Badge className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white">
                          {asPct(fusedScore)}
                        </Badge>
                      </div>
                      <div className="text-xs text-slate-600 mt-1">
                        Method: {m?.ensemble?.method ?? 'fused_weighted_average'} • Models:{' '}
                        {m?.ensemble?.models_used ?? perModel.length}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Found Labels (unique) */}
          {foundLabels && foundLabels.length > 0 && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Detected Features</h4>
              <div className="flex flex-wrap gap-2">
                {foundLabels.map((label, idx) => (
                  <Badge
                    key={`${label}-${idx}`}
                    variant="outline"
                    className="capitalize bg-white/70 border-slate-300 text-slate-700 hover:bg-slate-100 font-medium"
                  >
                    {label}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* YOLO detections — compact summary instead of long list */}
          {groupedDetections.length > 0 && (
            <div>
              <h4 className="font-semibold mb-3 text-slate-800">Object Detections</h4>
              <div className="flex flex-wrap gap-2">
                {groupedDetections.map(({ label, count }) => (
                  <span
                    key={label}
                    className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs bg-white/70"
                  >
                    <span className="capitalize">{label}</span>
                    <span className="opacity-70 ml-1">×{count}</span>
                  </span>
                ))}
              </div>

              {/* If you ever need the verbose per-box list, flip this to true */}
              {false && (
                <div className="space-y-2 mt-3">
                  {detections.map((d, i) => (
                    <div key={i} className="bg-white/70 p-3 rounded-lg border border-slate-200">
                      <div className="flex justify-between items-center">
                        <span className="capitalize font-medium">{d.label ?? 'object'}</span>
                        {typeof d.conf === 'number' && (
                          <Badge variant="outline">{asPct(d.conf)} confidence</Badge>
                        )}
                      </div>
                      {Array.isArray(d.box) && d.box.length >= 4 && (
                        <div className="text-xs text-slate-600 mt-1">
                          Box: [{d.box.map((n) => Math.round(n)).join(', ')}]
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <Separator className="bg-slate-200" />

          {/* Images */}
          <div className="space-y-6">
            {origImg && (
              <div>
                <h4 className="font-semibold mb-3 text-slate-800">Original Image</h4>
                <div className="relative group">
                  <img
                    src={withBase(origImg)}
                    alt="Original"
                    className="w-full max-h-64 object-contain rounded-lg border-2 border-slate-200 bg-white shadow-lg transition-transform duration-300 group-hover:scale-105"
                    loading="lazy"
                  />
                </div>
              </div>
            )}

            {(visImg || visualizationImage) && (
              <div>
                <h4 className="font-semibold mb-3 text-slate-800">Analysis Visualization</h4>
                <div className="relative group">
                  <img
                    src={withBase(visImg ?? visualizationImage!)}
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
