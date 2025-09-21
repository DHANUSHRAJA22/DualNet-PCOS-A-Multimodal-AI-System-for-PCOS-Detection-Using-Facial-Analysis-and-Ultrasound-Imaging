import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Stethoscope, 
  Zap, 
  Brain, 
  ScanLine, 
  Users, 
  Shield,
  ChevronDown,
  CheckCircle,
  Lock,
  Sparkles,
  AlertTriangle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { UploadArea } from '@/components/UploadArea'
import { MedicalDisclaimer } from '@/components/MedicalDisclaimer'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { BackendStatus } from '@/components/BackendStatus'
import { SampleImages } from '@/components/SampleImages'
import { resolveApiBase, pingHealth, postPredict, convertToLegacyFormat, isStructuredResponse, type HealthStatus } from '@/lib/api'
import { toast } from 'sonner'
import type { ProcessedImage } from '@/lib/image'

const Reveal = ({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.6, delay }}
  >
    {children}
  </motion.div>
)

export function IndexPage() {
  const navigate = useNavigate()
  const [faceImage, setFaceImage] = useState<ProcessedImage | null>(null)
  const [xrayImage, setXrayImage] = useState<ProcessedImage | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [backendStatus, setBackendStatus] = useState<HealthStatus>('unreachable')
  const [showDiagnostics, setShowDiagnostics] = useState(false)

  const hasImages = Boolean(faceImage || xrayImage)

  useEffect(() => {
    // Initial health check
    pingHealth().then(setBackendStatus)
  }, [])

  const showDiagnosticsDialog = () => {
    const apiBase = resolveApiBase()
    const isInSandbox = window.location.hostname !== 'localhost' && !window.location.protocol.startsWith('https')
    
    toast.error(
      <div className="space-y-2">
        <div className="font-medium">Backend Connection Failed</div>
        <div className="text-sm space-y-1">
          <div>Detected API base: <code className="bg-muted px-1 rounded">{apiBase || 'relative (proxy)'}</code></div>
          {isInSandbox && (
            <div className="text-orange-600">
              Running in sandbox - local proxy unavailable
            </div>
          )}
        </div>
      </div>,
      { duration: 8000 }
    )
    
    setShowDiagnostics(true)
  }

  const handleSampleSelect = (type: 'face' | 'xray', processedImage: ProcessedImage) => {
    if (type === 'face') {
      setFaceImage(processedImage)
      toast.success('Face sample image loaded successfully')
    } else {
      setXrayImage(processedImage)
      toast.success('X-ray sample image loaded successfully')
    }
  }

  const handleAnalyze = async () => {
    if (!hasImages) {
      toast.error('Please select at least one image to analyze')
      return
    }

    // Validate file sizes before sending
    const maxSize = 5 * 1024 * 1024 // 5MB
    if (faceImage && faceImage.size > maxSize) {
      toast.error(`Face image is too large (${(faceImage.size / 1024 / 1024).toFixed(1)}MB). Maximum size is 5MB.`)
      return
    }
    if (xrayImage && xrayImage.size > maxSize) {
      toast.error(`X-ray image is too large (${(xrayImage.size / 1024 / 1024).toFixed(1)}MB). Maximum size is 5MB.`)
      return
    }
    // Pre-check backend health
    const currentStatus = await pingHealth()
    setBackendStatus(currentStatus)
    
    if (currentStatus !== 'online') {
      toast.error('Backend is not available. Please check your connection.')
      setShowDiagnostics(true)
      return
    }

    setIsAnalyzing(true)
    
    try {
      const formData = new FormData()
      
      if (faceImage) {
        formData.append('face_img', faceImage.file)
      }
      
      if (xrayImage) {
        formData.append('xray_img', xrayImage.file)
      }

      const results = await postPredict(formData, true) // Use structured format
      
      // Convert to legacy format for existing Results page
      const legacyResults = isStructuredResponse(results) 
        ? convertToLegacyFormat(results)
        : results
      
      navigate('/results', { state: { results: results } })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Analysis failed'
      
      // Check for backend error format
      if (message.includes('{"ok":false')) {
        try {
          const errorData = JSON.parse(message.substring(message.indexOf('{"ok":false')))
          toast.error(`Analysis failed: ${errorData.details || errorData.message || 'Unknown error'}`)
        } catch {
          toast.error(`Analysis failed: ${message}`)
        }
      } else if (message.includes('ECONNREFUSED') || message.includes('fetch') || message.includes('NetworkError')) {
        showDiagnosticsDialog()
      } else {
        toast.error(`Analysis failed: ${message}`)
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-[#D9F3F2] to-[#F3E9FB] relative overflow-hidden">
        <div className="container mx-auto px-4 py-16">
          <div className="max-w-4xl mx-auto text-center py-12">
            <Reveal>
              <Badge className="mb-8 bg-white/80 text-slate-700 border-slate-200 hover:bg-white shadow-lg backdrop-blur-sm animate-float">
                AI-Powered Medical Analysis
              </Badge>
            </Reveal>
            
            <Reveal delay={0.1}>
              <h1 className="text-5xl md:text-7xl font-bold mb-8 leading-tight">
                <span className="text-[#1E293B]">Multimodal PCOS</span>
                <br />
                <span className="bg-gradient-to-r from-[#2DD4BF] to-[#3B82F6] bg-clip-text text-transparent font-extrabold">
                  Analyzer
                </span>
              </h1>
            </Reveal>
            
            <Reveal delay={0.2}>
              <p className="text-xl md:text-2xl text-[#64748B] mb-12 max-w-3xl mx-auto leading-relaxed font-medium">
                Advanced AI technology for early PCOS detection using facial recognition and X-ray analysis. 
                Upload your images for instant, comprehensive screening.
              </p>
            </Reveal>
            
            <Reveal delay={0.3}>
              <div className="flex flex-wrap justify-center gap-6 mb-8">
                {[
                  { text: 'Deep Learning', color: 'bg-[#14B8A6]' },
                  { text: 'Instant Results', color: 'bg-[#0EA5E9]' },
                  { text: 'Privacy First', color: 'bg-[#8B5CF6]' }
                ].map((feature, index) => (
                  <Badge 
                    key={feature.text}
                    className={`${feature.color} text-white px-6 py-3 text-base font-semibold rounded-full shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300 hover:brightness-110`}
                  >
                    {feature.text}
                  </Badge>
                ))}
              </div>
            </Reveal>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-6xl mx-auto space-y-16">
          {/* Upload Section */}
          <Reveal>
            <Card className="shadow-vibrant-lg border-0 card-gradient backdrop-blur-sm">
              <CardHeader className="text-center pb-6">
                <CardTitle className="text-3xl mb-4 text-gradient bg-gradient-to-r from-purple-600 via-indigo-600 to-teal-600 bg-clip-text text-transparent">
                  Upload Your Images
                </CardTitle>
                <CardDescription className="text-lg max-w-2xl mx-auto text-slate-600">
                  Upload a facial photo and/or uterus X-ray for comprehensive PCOS analysis. 
                  You can upload one or both images.
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-8">
                <BackendStatus onStatusChange={setBackendStatus} />
                
                {backendStatus === 'unreachable' && (
                  <Alert className="border-orange-200 bg-orange-50">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      Backend is unreachable. If running in a sandbox environment, please set up a public backend URL using the "Set API URL" button above.
                    </AlertDescription>
                  </Alert>
                )}
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <UploadArea
                    id="face-upload"
                    label="Face Image"
                    subtext="Upload a clear facial photo for symptom analysis"
                    tips="Use a neutral expression, front-facing, no sunglasses."
                    onChange={setFaceImage}
                  />
                  
                  <UploadArea
                    id="xray-upload"
                    label="Uterus X-ray"
                    subtext="Upload a uterus X-ray for morphological analysis"
                    tips="Standard pelvic X-ray. Upload only clear, high-contrast images."
                    onChange={setXrayImage}
                  />
                </div>
                
                <div className="text-center space-y-4">
                  <Button 
                    onClick={handleAnalyze}
                    disabled={!hasImages || isAnalyzing || backendStatus !== 'online'}
                    size="lg"
                    className="px-12 py-6 text-lg bg-gradient-to-r from-purple-600 via-indigo-600 to-teal-600 hover:from-purple-700 hover:via-indigo-700 hover:to-teal-700 shadow-vibrant transition-all duration-300 hover:scale-105 hover:shadow-vibrant-lg"
                  >
                    {isAnalyzing ? (
                      <>
                        <LoadingSpinner size="sm" className="mr-2" />
                        Analyzing Images...
                      </>
                    ) : (
                      <>
                        <Zap className="h-5 w-5 mr-2" />
                        Analyze Images
                      </>
                    )}
                  </Button>
                  
                  {!hasImages && (
                    <p className="text-sm text-muted-foreground">
                      Please upload at least one image to proceed
                    </p>
                  )}
                  
                  {backendStatus !== 'online' && hasImages && (
                    <p className="text-sm text-amber-600 font-medium">
                      Backend must be online to analyze images
                    </p>
                  )}
                </div>

                <MedicalDisclaimer />
                
                <SampleImages onSelectSample={handleSampleSelect} />
              </CardContent>
            </Card>
          </Reveal>

          {/* How It Works Section */}
          <Reveal>
            <div className="text-center space-y-12">
              <div>
                <h2 className="text-4xl font-bold mb-4 text-gradient bg-gradient-to-r from-purple-600 via-indigo-600 to-teal-600 bg-clip-text text-transparent">
                  How It Works
                </h2>
                <p className="text-xl text-slate-600 max-w-2xl mx-auto">
                  Our AI system uses advanced deep learning models to analyze multiple indicators
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {[
                  {
                    icon: Brain,
                    title: 'Facial Analysis',
                    description: 'AI analyzes facial features for visible PCOS symptoms like hirsutism and skin changes',
                    delay: 0,
                    color: 'from-purple-500 to-indigo-500'
                  },
                  {
                    icon: ScanLine,
                    title: 'X-ray Detection',
                    description: 'YOLOv8 object detection identifies morphological features in uterus X-rays',
                    delay: 0.1,
                    color: 'from-indigo-500 to-teal-500'
                  },
                  {
                    icon: Sparkles,
                    title: 'Combined Results',
                    description: 'Intelligent fusion of both analyses provides comprehensive risk assessment',
                    delay: 0.2,
                    color: 'from-teal-500 to-emerald-500'
                  }
                ].map((item, index) => (
                  <Reveal key={index} delay={item.delay}>
                    <Card className="h-full card-gradient hover:shadow-vibrant-lg transition-all duration-300 hover:-translate-y-2 hover:scale-105 border-0">
                      <CardContent className="p-8 text-center">
                        <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br ${item.color} mb-6 shadow-vibrant`}>
                          <item.icon className="h-8 w-8 text-white" />
                        </div>
                        <h3 className="text-xl font-semibold mb-4 text-slate-800">{item.title}</h3>
                        <p className="text-slate-600 leading-relaxed">{item.description}</p>
                      </CardContent>
                    </Card>
                  </Reveal>
                ))}
              </div>
            </div>
          </Reveal>

          {/* FAQ Section */}
          <Reveal>
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-4xl font-bold mb-4 text-gradient bg-gradient-to-r from-purple-600 via-indigo-600 to-teal-600 bg-clip-text text-transparent">
                  Frequently Asked Questions
                </h2>
                <p className="text-xl text-slate-600">
                  Everything you need to know about our PCOS screening tool
                </p>
              </div>
              
              <Accordion type="single" collapsible className="space-y-4 max-w-3xl mx-auto">
                {[
                  {
                    question: "How accurate is the AI analysis?",
                    answer: "Our AI models are trained on extensive medical datasets and achieve high accuracy rates. However, this tool is for screening purposes only and should not replace professional medical diagnosis. Always consult with healthcare professionals for definitive diagnosis."
                  },
                  {
                    question: "Is my data secure and private?",
                    answer: "Yes, your privacy is our top priority. Images are processed locally when possible, and any data transmitted is encrypted. We do not store your images permanently, and all processing is done for analysis purposes only."
                  },
                  {
                    question: "What types of images should I upload?",
                    answer: "For facial analysis, upload a clear, front-facing photo with neutral expression and good lighting. For X-ray analysis, use standard pelvic X-ray images with good contrast. Supported formats include JPEG, PNG, and WebP."
                  },
                  {
                    question: "How should I interpret the results?",
                    answer: "Results are presented as risk assessments (Low, Moderate, High) based on AI analysis. These are screening indicators only and should be discussed with a qualified healthcare professional for proper medical interpretation."
                  },
                  {
                    question: "Can I use smartphone photos?",
                    answer: "Yes, smartphone photos can be used for facial analysis if they are clear, well-lit, and show the face directly. However, professional medical imaging is recommended for X-ray analysis for best results."
                  },
                  {
                    question: "What image formats are supported?",
                    answer: "We support JPEG, PNG, and WebP image formats. The system automatically handles image orientation and optimization for analysis."
                  },
                  {
                    question: "Will my images be stored?",
                    answer: "No, your images are not permanently stored. They are processed for analysis and then discarded. We prioritize your privacy and data security."
                  },
                  {
                    question: "Does this replace lab tests?",
                    answer: "No, this tool is for screening and educational purposes only. It does not replace laboratory tests, clinical examinations, or professional medical diagnosis. Always consult healthcare professionals for comprehensive evaluation."
                  },
                  {
                    question: "What should I do after a high-risk result?",
                    answer: "If you receive a high-risk assessment, consult with a qualified healthcare professional immediately. They can perform proper diagnostic tests and provide appropriate medical guidance based on your individual situation."
                  }
                ].map((faq, index) => (
                  <AccordionItem key={index} value={`item-${index}`} className="card-gradient border-0 rounded-lg px-6 shadow-vibrant">
                    <AccordionTrigger className="text-left hover:no-underline py-6">
                      <span className="font-semibold text-slate-800">{faq.question}</span>
                    </AccordionTrigger>
                    <AccordionContent className="text-slate-600 leading-relaxed pb-6">
                      {faq.answer}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </div>
          </Reveal>
        </div>
      </div>

      {/* Diagnostics Dialog */}
      <Dialog open={showDiagnostics} onOpenChange={setShowDiagnostics}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-orange-500" />
              Backend Connection Issue
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="bg-muted/50 p-4 rounded-lg space-y-2">
              <div className="text-sm">
                <strong>Detected API Base:</strong> 
                <code className="ml-2 bg-background px-2 py-1 rounded text-xs">
                  {resolveApiBase() || 'relative (proxy)'}
                </code>
              </div>
            </div>
            
            <div className="space-y-3 text-sm">
              <p className="font-medium">If running in StackBlitz or other sandbox:</p>
              <ol className="list-decimal list-inside space-y-1 ml-4">
                <li>Start your Flask backend locally</li>
                <li>Expose it publicly with ngrok: <code className="bg-muted px-1 rounded">ngrok http 5000</code></li>
                <li>Or deploy to Render/Railway/Heroku</li>
                <li>Open this app with: <code className="bg-muted px-1 rounded">?api=https://your-backend-url</code></li>
              </ol>
              
              <p className="mt-4">
                <strong>Quick test:</strong> Open <code className="bg-muted px-1 rounded">{resolveApiBase() || window.location.origin}/health</code> in your browser - it should return JSON with status "ok".
              </p>
            </div>
            
            <Button onClick={() => setShowDiagnostics(false)} className="w-full">
              Got it
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Footer */}
      <footer className="bg-gradient-to-br from-slate-900 via-purple-900 to-indigo-900 text-white py-12 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-indigo-500/10 to-teal-500/10"></div>
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center space-y-6 relative z-10">
            <div className="flex items-center justify-center gap-2 mb-4">
              <Stethoscope className="h-6 w-6 text-emerald-400" />
              <span className="text-xl font-semibold">Multimodal PCOS Analyzer</span>
            </div>
            
            <div className="space-y-2 text-slate-200">
              <p>For educational and research purposes</p>
              <p>This tool does not provide medical advice. Consult healthcare professionals for diagnosis.</p>
              <p className="font-medium">Multimodal PCOS Analyzer - AI-powered screening tool</p>
              <p className="text-sm">Project by <span className="font-medium text-emerald-400">DHANUSH RAJA (21MIC0158)</span></p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}