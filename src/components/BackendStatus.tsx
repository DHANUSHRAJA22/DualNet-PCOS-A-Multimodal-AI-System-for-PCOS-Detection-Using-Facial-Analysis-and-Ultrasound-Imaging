import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle, AlertCircle, XCircle, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { pingHealth, resolveApiBase, getEnhancedHealth, type HealthStatus } from '@/lib/api'
import { toast } from 'sonner'

interface BackendStatusProps {
  onStatusChange?: (status: HealthStatus) => void
}

export function BackendStatus({ onStatusChange }: BackendStatusProps) {
  const [status, setStatus] = useState<HealthStatus>('unreachable')
  const [isChecking, setIsChecking] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [apiUrl, setApiUrl] = useState('')
  const [healthDetails, setHealthDetails] = useState<any>(null)

  const checkHealth = async () => {
    setIsChecking(true)
    try {
      const healthStatus = await pingHealth()
      setStatus(healthStatus)
      onStatusChange?.(healthStatus)
      
      // Get detailed health info if online
      if (healthStatus === 'online') {
        try {
          const details = await getEnhancedHealth()
          setHealthDetails(details)
          
          // Check for missing X-ray models
          const xrayModels = Object.keys(details.models || {}).filter(key => 
            key.includes('xray') && details.models[key].lazy_loadable
          )
          
          if (xrayModels.length === 0) {
            toast.warning('No X-ray models available - only facial analysis will work', { duration: 6000 })
          }
        } catch (error) {
          console.warn('Could not fetch detailed health info:', error)
        }
      }
    } catch (error) {
      setStatus('unreachable')
      onStatusChange?.('unreachable')
      setHealthDetails(null)
    } finally {
      setIsChecking(false)
    }
  }

  useEffect(() => {
    checkHealth()
    setApiUrl(resolveApiBase())
    
    // Check health periodically
    const interval = setInterval(checkHealth, 30000) // Every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const handleSetApiUrl = () => {
    if (!apiUrl.trim()) {
      toast.error('Please enter a valid API URL')
      return
    }

    // Update URL with new api parameter
    const url = new URL(window.location.href)
    if (apiUrl.trim()) {
      url.searchParams.set('api', apiUrl.trim())
    } else {
      url.searchParams.delete('api')
    }
    
    // Update browser URL without reload
    window.history.pushState({}, '', url.toString())
    
    setDialogOpen(false)
    toast.success('API URL updated')
    
    // Recheck health with new URL
    setTimeout(checkHealth, 100)
  }

  const getStatusConfig = () => {
    switch (status) {
      case 'online':
        return {
          icon: CheckCircle,
          label: 'Online',
          variant: 'default' as const,
          className: 'bg-green-100 text-green-800 border-green-200 hover:bg-green-200'
        }
      case 'offline':
        return {
          icon: XCircle,
          label: 'Offline',
          variant: 'destructive' as const,
          className: 'bg-red-100 text-red-800 border-red-200 hover:bg-red-200'
        }
      case 'unreachable':
        return {
          icon: AlertCircle,
          label: 'Unreachable',
          variant: 'secondary' as const,
          className: 'bg-orange-100 text-orange-800 border-orange-200 hover:bg-orange-200'
        }
    }
  }

  const config = getStatusConfig()
  const StatusIcon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-2 mb-4"
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-slate-600">Backend Status:</span>
        <Badge className={`${config.className} shadow-sm transition-all duration-300 hover:scale-105`}>
          {isChecking ? (
            <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current mr-1" />
          ) : (
            <StatusIcon className="h-3 w-3 mr-1" />
          )}
          {isChecking ? 'Checking...' : config.label}
        </Badge>
        
        {healthDetails && status === 'online' && (
          <Badge variant="outline" className="text-xs">
            v{healthDetails.version} • {Math.round(healthDetails.uptime_seconds)}s uptime
            {healthDetails.config?.use_ensemble && (
              <span className="ml-1">• Ensemble</span>
            )}
          </Badge>
        )}
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogTrigger asChild>
          <Button variant="ghost" size="sm" className="h-6 px-2 text-slate-600 hover:text-purple-600 hover:bg-purple-50 transition-colors">
            <Settings className="h-3 w-3 mr-1" />
            Set API URL
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-gradient bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Configure Backend API
            </DialogTitle>
              <div className="text-xs text-muted-foreground">
                Default: http://localhost:8000 (FastAPI backend)
              </div>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="api-url">Backend API URL</Label>
              <Input
                id="api-url"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="https://your-backend.com"
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                <strong>Quick test:</strong> Open <code className="bg-muted px-1 rounded">{resolveApiBase() || 'http://localhost:8000'}/health</code> in your browser - it should return JSON with model status.
              </p>
              <ul className="text-xs text-muted-foreground mt-2 space-y-1">
                <li>Start your FastAPI backend locally on port 8000</li>
                <li>Expose it publicly with ngrok: <code className="bg-muted px-1 rounded">ngrok http 8000</code></li>
              </ul>
            </div>
            <div className="flex gap-2">
              <Button 
                onClick={handleSetApiUrl} 
                className="flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-vibrant"
              >
                Update API URL
              </Button>
              <Button 
                variant="outline" 
                onClick={checkHealth}
                className="border-purple-200 text-purple-600 hover:bg-purple-50"
              >
                Test Connection
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  )
}