# Frontend — Fleet Ops Dashboard

React + TypeScript + Vite. See the [root README](../README.md) for the full
project overview, architecture, and setup instructions.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api to the backend on :8000
npm run build    # type-check + production build
npm run lint
```

Key files:
- `src/App.tsx` — the dashboard (fleet list, RUL trajectory, sensor telemetry,
  risk/maintenance panels)
- `src/index.css` — design tokens and component styles
- `vite.config.ts` — dev server + `/api` proxy config
