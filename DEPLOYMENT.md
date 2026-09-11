# GeoVision production deployment

GeoVision uses two services:

- **Vercel** hosts the Next.js frontend in `sar-colorization-app/frontend`.
- **Render** hosts the FastAPI inference service and loads the three PyTorch
  checkpoints. The model files are intentionally excluded from Git.

The frontend must be built with `NEXT_PUBLIC_API_URL` set to the public Render
service URL. The backend must allow the Vercel frontend URL through
`CORS_ORIGINS`.

## Local development

Use Python 3.11 or newer. From the repository root, create an environment and
install the FastAPI runtime:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r sar-colorization-app/requirements-backend.txt
```

Copy the environment template and set any optional AI-provider values:

```bash
cp sar-colorization-app/.env.example sar-colorization-app/.env
```

The default checkpoint locations are:

```text
pix2pix_gen_180.pth
models/checkpoints/sarfusionformer_256_decoder_best.pt
models/checkpoints/color_corrector_256_best.pt
```

Start the API from the repository root. This command intentionally includes
`--app-dir` so the `sar-colorization-app` directory never needs to become a
Python package or be renamed.

```bash
.venv/bin/uvicorn backend:app --app-dir sar-colorization-app --host 127.0.0.1 --port 8010
```

In a second terminal, start the frontend:

```bash
cd sar-colorization-app/frontend
npm ci
npm run dev
```

Set `NEXT_PUBLIC_API_URL=http://127.0.0.1:8010` in
`sar-colorization-app/frontend/.env.local` for local frontend development.

## Checkpoint storage and download

Do not put checkpoints in GitHub or Git LFS. The largest file is about 208 MiB,
so it exceeds GitHub's regular file limit, and model weights are better managed
independently from application source. Use a private Hugging Face **model**
repository, for example `your-hf-username/geovision-checkpoints`.

Upload the files at the root of that Hugging Face repository, retaining these
exact filenames:

```bash
hf auth login
hf repo create your-hf-username/geovision-checkpoints --repo-type model --private
hf upload your-hf-username/geovision-checkpoints ./pix2pix_gen_180.pth pix2pix_gen_180.pth
hf upload your-hf-username/geovision-checkpoints ./models/checkpoints/sarfusionformer_256_decoder_best.pt sarfusionformer_256_decoder_best.pt
hf upload your-hf-username/geovision-checkpoints ./models/checkpoints/color_corrector_256_best.pt color_corrector_256_best.pt
```

Create a Hugging Face token with **Read** access for Render. The script below
downloads missing checkpoints, keeps existing non-empty files, verifies the
download is non-empty and that its SHA-256 matches the downloaded Hub artifact,
and fails with a clear message if it cannot reach the repository or token lacks
access:

```bash
HF_MODEL_REPO=your-hf-username/geovision-checkpoints \
HF_TOKEN=hf_your_read_token \
python scripts/download_checkpoints.py
```

If you store a remote filename in a subfolder, set the corresponding
`*_REMOTE_FILENAME` variable. If local checkpoint locations differ, set their
`*_CHECKPOINT` variables to absolute paths or paths relative to the repository
root. The FastAPI service and download script use the same resolution rules.

## Render deployment

The repository-root `render.yaml` defines a Python web service named
`geovision-api`. Keep the Render service root at the repository root because
the backend imports `src/` from there. Its build installs
`sar-colorization-app/requirements-backend.txt`, downloads checkpoints, then
starts FastAPI with:

```bash
uvicorn backend:app --app-dir sar-colorization-app --host 0.0.0.0 --port $PORT
```

### Required Render environment variables

Set these in the initial Blueprint flow or the Render dashboard:

```text
HF_MODEL_REPO=your-hf-username/geovision-checkpoints
HF_TOKEN=hf_your_read_token                 # required for a private/gated repository
CORS_ORIGINS=https://geo-vision-main.vercel.app
```

Set checkpoint paths only if using non-default locations:

```text
PIX2PIX_CHECKPOINT=pix2pix_gen_180.pth
SARFUSIONFORMER_CHECKPOINT=models/checkpoints/sarfusionformer_256_decoder_best.pt
COLOR_CORRECTOR_CHECKPOINT=models/checkpoints/color_corrector_256_best.pt
```

The Blueprint already supplies the default remote filenames. Add a server-side
`GEMINI_API_KEY` only if the optional image-analysis provider is required.
Never expose it through a `NEXT_PUBLIC_*` variable.

### Deploy on Render

1. Push the source changes to the Git branch used for deployment. Do **not**
   add checkpoint files or secrets.
2. In Render, select **New → Blueprint**, connect the repository, and use the
   root `render.yaml`.
3. Enter `HF_MODEL_REPO`, `HF_TOKEN` (for a private repository), and the exact
   Vercel origin as `CORS_ORIGINS`.
4. Deploy. The build log must show the checkpoint downloader finishing for all
   three files.
5. Open `https://<render-service>.onrender.com/health`. Each model needs
   `available: true` before connecting the frontend.

The provided `standard` plan is a baseline, not a performance guarantee. This
is CPU inference unless a GPU-capable hosting configuration is selected. If
startup or inference runs out of memory, increase the Render instance size;
do not reduce the models or checkpoints.

## Vercel deployment

1. Set Vercel's **Root Directory** to `sar-colorization-app/frontend` and keep
   the **Next.js** framework preset.
2. In Vercel → Settings → Environment Variables, set this for Production (and
   Preview if desired):

   ```text
   NEXT_PUBLIC_API_URL=https://<render-service>.onrender.com
   ```

   Do not add a trailing slash.
3. Redeploy Vercel after changing that variable. `NEXT_PUBLIC_API_URL` is
   embedded into the client bundle during the Next.js build.
4. Upload a test image. The model-status panel should turn online after the
   browser can call the Render `/health` endpoint.

## Troubleshooting

| Symptom | Cause and resolution |
| --- | --- |
| Vercel UI says `Failed to fetch` and models are offline | `NEXT_PUBLIC_API_URL` is missing, has the local `127.0.0.1` value, or the Vercel frontend was not redeployed. |
| Browser reports a CORS error | Add the exact Vercel origin, for example `https://geo-vision-main.vercel.app`, to Render's comma-separated `CORS_ORIGINS`, then redeploy Render. |
| Render build fails in `download_checkpoints.py` | Check the Hugging Face repository name, remote filenames, and the Read token for a private/gated repository. |
| `/health` lists a model as unavailable | Confirm that its checkpoint environment path matches the downloaded location and review the model-specific error returned by `/health`. |
| Render health check never completes | Check RAM and startup logs. All models load before the application starts; use a larger instance instead of changing model logic. |
| Gemini settings cannot persist on Render | Use server-side `GEMINI_API_KEY`; macOS Keychain storage is intentionally local-only. |

## Security checklist

- Keep the Hugging Face model repository private unless the weights may be public.
- Store `HF_TOKEN`, `GEMINI_API_KEY`, and encryption keys only in Render/Vercel
  environment-variable settings, never in source files.
- Restrict `CORS_ORIGINS` to your actual frontend domains.
- Do not upload checkpoints to GitHub, commit them, or add them to Git LFS.
