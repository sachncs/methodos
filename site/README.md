# methodos · site

Premium product page for [methodos](https://github.com/sachncs/methodos),
deployed to GitHub Pages.

- **Framework:** React 18 + TypeScript + Vite
- **Styling:** Tailwind CSS with a custom dark, cinematic design system
- **Motion:** Framer Motion (subtle, restrained)
- **Output:** Static site under `dist/`, served from `/methodos/`

## Develop

```bash
cd site
npm install
npm run dev
```

## Build

```bash
npm run build
```

The production build is emitted to `site/dist/`.

## Deploy

Pushing to `master` triggers `.github/workflows/pages.yml`, which builds
`site/` and publishes `site/dist/` to GitHub Pages at
`https://sachncs.github.io/methodos/`.
