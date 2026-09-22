# JIMBO
Judgment-induced model for basketball outcomes: NBA 2026-27 projections, updated nightly.
- `pipeline/` data fetch (runs on GitHub Actions) and live parsing
- `model/` JIMBO model code
- `live/` fetched NBA data (schedule, play-by-play, box scores, injuries, lines)

## Phone app
https://jvector26.github.io/JIMBO/ (GitHub Pages, deployed from `site/` by `.github/workflows/pages.yml` after each nightly run).
Add it to your home screen: Safari > Share > Add to Home Screen (Android Chrome: menu > Install app).
- `site/index.html`, `sw.js`, `manifest.webmanifest`, `icons/`, `data/dash.json` are built by Claude (`code/build_app.py`
  from the dashboard template) and change only when the model changes.
- `site/data/latest.json`, `history.json` are written by the nightly run.
- Player edits in the app save on the device; with a GitHub token saved in the app's Settings they are written to
  `overrides.json`, which the nightly run applies. Token: fine-grained, this repository only, Contents read and write.
