import { useRouteError, isRouteErrorResponse } from 'react-router-dom'
import { AlertTriangle, Home, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function AppErrorBoundary() {
  const error = useRouteError()

  const handleReload = () => {
    window.location.reload()
  }

  const handleGoHome = () => {
    window.location.href = '/'
  }

  if (isRouteErrorResponse(error)) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center p-4">
        <Card className="w-full max-w-md shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-600">
              <AlertTriangle className="w-5 h-5" />
              Navigation Error
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <p className="text-lg font-semibold">Status: {error.status}</p>
              <p className="text-sm text-muted-foreground">{error.statusText}</p>
            </div>
            
            {error.data && (
              <div className="bg-muted p-3 rounded-md">
                <pre className="text-xs overflow-auto">{String(error.data)}</pre>
              </div>
            )}
            
            <div className="flex gap-2">
              <Button onClick={handleGoHome} className="flex-1">
                <Home className="w-4 h-4 mr-2" />
                Go Home
              </Button>
              <Button onClick={handleReload} variant="outline" className="flex-1">
                <RefreshCw className="w-4 w-4 mr-2" />
                Reload
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (error instanceof Error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center p-4">
        <Card className="w-full max-w-md shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-600">
              <AlertTriangle className="w-5 h-5" />
              Application Error
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <p className="text-lg font-semibold">Something went wrong</p>
              <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
            </div>
            
            {process.env.NODE_ENV === 'development' && error.stack && (
              <div className="bg-muted p-3 rounded-md">
                <pre className="text-xs overflow-auto">{error.stack}</pre>
              </div>
            )}
            
            <div className="flex gap-2">
              <Button onClick={handleGoHome} className="flex-1">
                <Home className="w-4 h-4 mr-2" />
                Go Home
              </Button>
              <Button onClick={handleReload} variant="outline" className="flex-1">
                <RefreshCw className="w-4 h-4 mr-2" />
                Reload
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <AlertTriangle className="w-5 h-5" />
            Unknown Error
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-center">
            <p className="text-lg font-semibold">An unknown error occurred</p>
            <p className="text-sm text-muted-foreground mt-2">
              Please try refreshing the page or returning to the home page.
            </p>
          </div>
          
          <div className="flex gap-2">
            <Button onClick={handleGoHome} className="flex-1">
              <Home className="w-4 h-4 mr-2" />
              Go Home
            </Button>
            <Button onClick={handleReload} variant="outline" className="flex-1">
              <RefreshCw className="w-4 h-4 mr-2" />
              Reload
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}