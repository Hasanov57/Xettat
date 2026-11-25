# Xəttat website

Static site (no build step): `index.html`, `style.css`, `app.js`, shared core in `shared/`.

Local preview (after `python scripts/pack_extension.py --model exported/<run>` so that `model/` and `vendor/` exist):

```bash
cd web && python -m http.server 8000   # open http://localhost:8000
```

Publishing: GitHub → Settings → Pages → deploy from branch, folder `/web` is not selectable, so use the
`.github/workflows/pages.yml` workflow (added when the final model is ready) or copy `web/` to a `gh-pages` branch.
