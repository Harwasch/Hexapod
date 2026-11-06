### Console render check (local, no training)

```sh
TMPDIR="$PWD/node_modules/.cache/codex-tmp" node apps/web/scripts/console-check.mjs \
  --cameras /path/to/cameras.json --ref-dir /path/to/references \
  --source /path/to/snapshot.spz --out /path/to/check --budget full
```

Run from the repository root with Playwright Chromium already installed. `--source` also
accepts a tileset directory. The script starts its own loopback Vite server (an ephemeral
port, or `E2E_PORT`), serves only the specified inputs, and blocks browser requests off
that server. `console-check.html` is a development harness, absent from Vite's production
inputs and `public/`; it and its modules do not ship in `pnpm build`.

The `train-*` entries of `cameras.json` supply native width/height, gsplat world-to-camera
`viewmat`, and `K`. Reference files are `trained-<entry>.png`. The solve frame is unchanged;
projection includes both focal lengths and the principal point. Rendering uses the console's
Spark setup and tiles streaming, black background and no overlays. Classic filtering is the
default for the existing references; use `--antialiased` for an antialiased training run.
Omitting `--budget` uses the device budget; a number sets it; `full` loads every leaf and
disables Spark LOD and the streaming cap. Both tile loading and Spark's `settled` state
must finish before capture; a timeout fails the check instead of scoring a partial view.

Each `console-train-*.png` and `scores.json` goes in `--out`, which must be separate from
inputs. PSNR uses **all pixels**, RGB bytes divided by 255, without gamma conversion,
masking, alignment or colour fitting (`null` means identical, infinite PSNR). These PNG
scores can differ slightly from the reference generator's pre-quantization float scores.
The report records the budget, filtering, drawn splat count and tile names. This is a
local browser check; it does not launch GPU training or use Modal.
