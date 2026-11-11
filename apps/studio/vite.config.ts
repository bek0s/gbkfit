// The page is built into the Python package (src/gbkfit_studio/page), with
// relative paths: the server serves it under a path of its own (/studio/).
// In development (npm run dev), the API goes to a server on this machine:
// the one of GBKFIT_SERVER (e.g. http://127.0.0.1:8766), or by default
// gbkfit-server --port 8765.
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// (the environment of Node, which runs this file; the page has no Node
// types)
declare const process: { env: Record<string, string | undefined> };

export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'src/gbkfit_studio/page', emptyOutDir: true },
  server: {
    proxy: { '/api': { target: process.env.GBKFIT_SERVER ?? 'http://127.0.0.1:8765', ws: true } },
  },
});
