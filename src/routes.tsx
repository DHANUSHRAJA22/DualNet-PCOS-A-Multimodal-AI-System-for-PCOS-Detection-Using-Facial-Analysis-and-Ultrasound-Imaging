import { createBrowserRouter } from 'react-router-dom';
import { Index } from './pages/Index';
import { Results } from './pages/Results';
import AppErrorBoundary from './components/AppErrorBoundary';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Index />,
    errorElement: <AppErrorBoundary />
  },
  {
    path: '/results',
    element: <Results />,
    errorElement: <AppErrorBoundary />
  }
], {
  future: {
    v7_relativeSplatPath: true,
    v7_fetcherPersist: true,
    v7_normalizeFormMethod: true,
    v7_partialHydration: true,
    v7_skipActionErrorRevalidation: true
  }
});