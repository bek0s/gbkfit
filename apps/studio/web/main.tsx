// The page of the Studio: its fonts and look, the cache of the answers of
// the server (refreshed by its events, see api/events.ts), and the Studio.
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-sans-condensed/600.css';
import './styles.css';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { takeToken } from './api/client';
import { App } from './App';

takeToken();

const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: 1 } } });

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
