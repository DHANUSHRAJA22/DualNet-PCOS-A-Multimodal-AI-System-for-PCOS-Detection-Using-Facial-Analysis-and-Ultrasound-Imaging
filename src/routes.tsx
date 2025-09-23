import { createBrowserRouter } from 'react-router-dom'
import { IndexPage } from './pages/Index'
import { Results } from './pages/Results'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <IndexPage />,
  },
  {
    path: '/results',
    element: <Results />,
  },
])