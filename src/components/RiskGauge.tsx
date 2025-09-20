import { motion } from 'framer-motion'
import { TrendingUp, AlertTriangle, CheckCircle, HelpCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface RiskGaugeProps {
  riskLevel: 'low' | 'moderate' | 'high' | 'unknown'
  confidence?: number
  className?: string
}

export function RiskGauge({ riskLevel, confidence = 0, className }: RiskGaugeProps) {
  const getRiskConfig = () => {
    switch (riskLevel) {
      case 'low':
        return {
          icon: CheckCircle,
          label: 'Low Risk',
          color: 'from-emerald-500 to-teal-500',
          bgColor: 'from-emerald-50 to-teal-50',
          borderColor: 'border-emerald-200',
          textColor: 'text-emerald-700',
          percentage: 25,
          description: 'Analysis shows minimal risk indicators'
        }
      case 'moderate':
        return {
          icon: AlertTriangle,
          label: 'Moderate Risk',
          color: 'from-amber-500 to-orange-500',
          bgColor: 'from-amber-50 to-orange-50',
          borderColor: 'border-amber-200',
          textColor: 'text-amber-700',
          percentage: 60,
          description: 'Some indicators warrant further evaluation'
        }
      case 'high':
        return {
          icon: TrendingUp,
          label: 'High Risk',
          color: 'from-rose-500 to-red-500',
          bgColor: 'from-rose-50 to-red-50',
          borderColor: 'border-rose-200',
          textColor: 'text-rose-700',
          percentage: 85,
          description: 'Multiple risk indicators detected'
        }
      default:
        return {
          icon: HelpCircle,
          label: 'Unknown',
          color: 'from-slate-400 to-gray-500',
          bgColor: 'from-slate-50 to-gray-50',
          borderColor: 'border-slate-200',
          textColor: 'text-slate-600',
          percentage: 0,
          description: 'Analysis pending or inconclusive'
        }
    }
  }

  const config = getRiskConfig()
  const Icon = config.icon

  return (
    <Card className={`border-2 ${config.borderColor} bg-gradient-to-br ${config.bgColor} ${className}`}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Icon className={`h-5 w-5 ${config.textColor}`} />
          Risk Assessment
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Risk Level Badge */}
        <div className="text-center">
          <Badge className={`text-lg px-6 py-3 bg-gradient-to-r ${config.color} text-white shadow-lg`}>
            {config.label}
          </Badge>
        </div>

        {/* Circular Gauge */}
        <div className="relative flex items-center justify-center">
          <div className="relative w-32 h-32">
            {/* Background Circle */}
            <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
              <circle
                cx="60"
                cy="60"
                r="50"
                fill="none"
                stroke="currentColor"
                strokeWidth="8"
                className="text-slate-200"
              />
              {/* Progress Circle */}
              <motion.circle
                cx="60"
                cy="60"
                r="50"
                fill="none"
                strokeWidth="8"
                strokeLinecap="round"
                className={`${config.textColor}`}
                style={{
                  strokeDasharray: `${2 * Math.PI * 50}`,
                }}
                initial={{ strokeDashoffset: 2 * Math.PI * 50 }}
                animate={{ 
                  strokeDashoffset: 2 * Math.PI * 50 * (1 - config.percentage / 100)
                }}
                transition={{ duration: 1.5, ease: "easeOut" }}
              />
            </svg>
            
            {/* Center Content */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.5, duration: 0.5 }}
                  className={`text-2xl font-bold ${config.textColor}`}
                >
                  {config.percentage}%
                </motion.div>
                <div className="text-xs text-slate-600">Risk Level</div>
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="text-center">
          <p className="text-sm text-slate-600 leading-relaxed">
            {config.description}
          </p>
        </div>

        {/* Confidence Indicator */}
        {confidence > 0 && (
          <div className="bg-white/70 p-3 rounded-lg border border-slate-200">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-700">AI Confidence</span>
              <span className="text-sm font-bold text-indigo-600">{confidence.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2">
              <motion.div
                className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${confidence}%` }}
                transition={{ duration: 1, delay: 0.5 }}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}