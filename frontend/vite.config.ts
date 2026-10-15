import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()], server: {port: 5173, strictPort: true, proxy: {'/api': process.env.FACETELLER_API || 'http://127.0.0.1:8767'}}, build: {sourcemap: false} });
