import { createBrowserRouter } from 'react-router-dom'
import { IndexPage } from './pages/Index'
import { Results } from './pages/Results'
import AppErrorBoundary from './components/AppErrorBoundary'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <IndexPage />,
    errorElement: <AppErrorBoundary />,
  },
  {
    path: '/results',
    element: <Results />,
    errorElement: <AppErrorBoundary />,
  },
  {
    path: '*',
    element: <AppErrorBoundary />,
  },
])