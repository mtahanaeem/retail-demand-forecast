import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/stores': 'http://localhost:8000',
      '/product-families': 'http://localhost:8000',
      '/forecast': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/model-info': 'http://localhost:8000',
    },
  },
});
