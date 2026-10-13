# methodos · site

Premium product page for [methodos](https://github.com/sachncs/methodos),
deployed to GitHub Pages.

- **Framework:** Astro (static output)
- **Styling:** Component-scoped CSS with a restrained product design system
- **Runtime:** Zero client-side JavaScript for the marketing page
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

## Browser smoke tests

After building, run the responsive Chromium checks locally:

```bash
npx playwright install chromium
npm run test:e2e
```

## Deploy

Pushing to `master` triggers `.github/workflows/pages.yml`, which builds
`site/` and publishes `site/dist/` to GitHub Pages at
`https://sachncs.github.io/methodos/`.
