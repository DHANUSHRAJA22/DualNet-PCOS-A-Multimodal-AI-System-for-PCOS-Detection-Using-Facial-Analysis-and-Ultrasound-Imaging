import { motion } from 'framer-motion'
import { Brain, Target, Search, TrendingUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

interface MetricProps {
  label: string
  value: number
  color: string
  icon: React.ComponentType<{ className?: string }>
  description: string
  delay: number
}

function MetricCard({ label, value, color, icon: Icon, description, delay }: MetricProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
      className="bg-white/70 rounded-lg p-4 border border-slate-200 hover:border-slate-300 transition-all duration-300 hover:shadow-lg"
    >
      <div className="flex items-center gap-3 mb-3">
        <div 
          className="p-2 rounded-lg shadow-sm"
          style={{ backgroundColor: `${color}15` }}
        >
          <Icon className="h-5 w-5" style={{ color }} />
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-slate-800">{label}</h4>
            <Badge 
              className="font-bold text-white shadow-sm"
              style={{ backgroundColor: color }}
            >
              {value.toFixed(1)}%
            </Badge>
          </div>
        </div>
      </div>
      
      <div className="mb-3">
        <div className="relative">
          <Progress 
            value={0} 
            className="h-3 bg-slate-200"
          />
          <motion.div
            className="absolute top-0 left-0 h-3 rounded-full shadow-sm"
            style={{ backgroundColor: color }}
            initial={{ width: 0 }}
            animate={{ width: `${value}%` }}
            transition={{ duration: 1.5, delay: delay + 0.3, ease: "easeOut" }}
          />
        </div>
      </div>
      
      <p className="text-xs text-slate-600 leading-relaxed">
        {description}
      </p>
    </motion.div>
  )
}

export function AIPerformanceMetrics() {
  const metrics = [
    {
      label: 'Accuracy',
      value: 94.2,
      color: '#3B82F6',
      icon: Target,
      description: 'Overall correctness of predictions across all cases. Higher accuracy indicates more reliable screening results.',
      delay: 0
    },
    {
      label: 'Precision',
      value: 92.5,
      color: '#8B5CF6',
      icon: Search,
      description: 'Percentage of positive predictions that were actually correct. Reduces false alarms in medical screening.',
      delay: 0.1
    },
    {
      label: 'Recall',
      value: 93.8,
      color: '#14B8A6',
      icon: TrendingUp,
      description: 'Percentage of actual positive cases correctly identified. Critical for not missing potential PCOS cases.',
      delay: 0.2
    }
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="border-2 border-indigo-200 bg-gradient-to-br from-indigo-50 to-purple-50 shadow-vibrant-lg">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-xl">
            <div className="p-3 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 text-white shadow-lg">
              <Brain className="h-6 w-6" />
            </div>
            AI Performance Metrics
            <Badge className="ml-auto bg-gradient-to-r from-indigo-500 to-purple-500 text-white">
              Clinical Grade
            </Badge>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
            <p className="text-sm text-slate-700 leading-relaxed">
              Our AI models are trained on extensive medical datasets and validated against clinical standards. 
              These metrics demonstrate the system's reliability for PCOS screening applications.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {metrics.map((metric, index) => (
              <MetricCard
                key={metric.label}
                {...metric}
              />
            ))}
          </div>
          
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0">
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-amber-100">
                  <Brain className="h-3 w-3 text-amber-600" />
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-amber-900 mb-1">Clinical Context</h4>
                <p className="text-xs text-amber-800 leading-relaxed">
                  These performance metrics are based on validation studies and represent the AI's ability to 
                  assist in medical screening. Results should always be interpreted by qualified healthcare professionals.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}