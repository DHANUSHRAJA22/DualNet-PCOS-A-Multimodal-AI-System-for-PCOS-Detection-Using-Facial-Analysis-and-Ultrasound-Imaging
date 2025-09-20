import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle, AlertCircle, XCircle, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { pingHealth, resolveApiBase, type HealthStatus } from '@/lib/api'
import { toast } from 'sonner'

interface BackendStatusProps {
  onStatusChange?: (status: HealthStatus) => void
}

export function BackendStatus({ onStatusChange }: BackendStatusProps) {
  const [status, setStatus] = useState<HealthStatus>('unreachable')
  const [isChecking, setIsChecking] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [apiUrl, setApiUrl] = useState('')

  const checkHealth = async () => {
    setIsChecking(true)
    try {
      const healthStatus = await pingHealth()
      setStatus(healthStatus)
      onStatusChange?.(healthStatus)
    } catch (error) {
      setStatus('unreachable')
      onStatusChange?.('unreachable')
    } finally {
      setIsChecking(false)
    }
  }

  useEffect(() => {
    checkHealth()
    setApiUrl(resolveApiBase())
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
                Leave empty to use local proxy (dev only)
              </p>
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