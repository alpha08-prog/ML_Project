# Frontend

React 19 + TypeScript frontend for the EEG classification project.

## Local development

```bash
cd frontend/ml
npm ci
npm run dev
```

Vite serves the app on `http://localhost:5173`.

## Quality checks

```bash
npm run lint
npm run format:check
npm run type-check
npm run build
```

## Dockerfiles

- `Dockerfile.dev` runs the Vite dev server for local Docker Compose usage.
- `Dockerfile` creates a production build and serves it through Nginx.

The production Nginx config also proxies backend routes so the deployed frontend can call `/api` without changing the app code.
