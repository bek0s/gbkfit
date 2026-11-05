// The page of the Studio: its look (the components of Mantine in the
// theme chosen, its fonts and colours, see theme.ts, in the scheme of the system unless
// another is chosen), the cache of the answers of the server (refreshed by
// its events, see api/events.ts), and the Studio.
import '@mantine/core/styles.css';
import './styles.css';
import { MantineProvider } from '@mantine/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StrictMode, useMemo, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { takeToken } from './api/client';
import { App } from './App';
import { mantineTheme, resolver, useTheme } from './theme';

takeToken();

const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: 1 } } });

// (Mantine, in the theme chosen, drawn again as another is)
function Themed({ children }: { children: ReactNode }) {
  const chosen = useTheme();
  const [theme, variables] = useMemo(() => [mantineTheme(chosen), resolver(chosen)], [chosen]);
  return (
    <MantineProvider theme={theme} cssVariablesResolver={variables} defaultColorScheme="auto">
      {children}
    </MantineProvider>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Themed>
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>
    </Themed>
  </StrictMode>,
);
