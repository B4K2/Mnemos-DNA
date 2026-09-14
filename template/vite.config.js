import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The comparison page is a plain .jsx file in this directory; this is only the shell
// that renders it. `npm run dev` -> http://localhost:5173
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: false },
});
