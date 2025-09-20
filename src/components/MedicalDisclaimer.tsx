import { AlertTriangle, Shield } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

export function MedicalDisclaimer() {
  return (
    <Card className="border-orange-200 bg-orange-50/50">
      <CardContent className="pt-6">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-orange-100">
              <AlertTriangle className="h-4 w-4 text-orange-600" />
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-orange-600" />
              <h3 className="font-semibold text-orange-900">Medical Disclaimer</h3>
            </div>
            <div className="text-sm text-orange-800">
              <p className="mb-2">
                This tool is for <strong>educational and research purposes only</strong>. 
                It does not provide medical advice and should not be used for diagnosis. 
                Always consult a qualified healthcare professional for proper medical evaluation and treatment.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}