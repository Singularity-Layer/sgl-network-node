# Singularity Node — Windows Beta

Run a Singularity grid node on Windows, serve real inference jobs, and earn — from
your own machine. This is a **private beta**: please don't reshare the download link.

## Before you start
- **Windows 10 or 11, 64-bit.**
- A **Solana wallet with ≥ 50,000 SGL staked** (stake at https://staking.x402layer.cc if
  you haven't). You'll approve a login in your browser with that wallet.
- A GPU helps a lot but isn't required (there's a CPU-only mode).
- ~3 GB free disk (llama.cpp + one small model).

## Fastest path (one click)
1. Put the `sgl.exe` you were given (or download it from your private link) into
   `%LOCALAPPDATA%\sgl-node\` — i.e. `C:\Users\<you>\AppData\Local\sgl-node\sgl.exe`.
2. Double-click **`sgl-beta-setup.bat`**.
3. Windows will warn *"Windows protected your PC"* (the app isn't code-signed yet) →
   click **More info → Run anyway**. This is expected for the beta.
4. The script installs llama.cpp, downloads a small model, opens your browser to **log in
   with your staked wallet**, then starts serving. Keep the window open.

## Manual path (if you prefer)
Open **PowerShell** in the folder with `sgl.exe`:
```powershell
.\sgl.exe version           # confirms it runs + prints the build hash
.\sgl.exe setup             # installs llama.cpp (add --cpu if you have no GPU)
# get a GGUF model, e.g. gemma-2-2b, and note its path
.\sgl.exe login --models gemma-2-2b        # browser login with your STAKED wallet
.\sgl.exe start --model-path C:\path\gemma-2-2b.gguf --model-name gemma-2-2b
```
Models you can serve today: `gemma-2-2b`, `qwen-2.5-7b`, `qwen-coder-7b`, `llama-3.1-8b`
(pick one your VRAM can hold; `gemma-2-2b` runs almost anywhere).

## Confirm it's working (the E2E check)
1. Open **https://cloud.x402compute.cc/network/console** with the same wallet.
2. Your machine should appear **Active** with a recent heartbeat.
3. Send it a job from **Grid → Chat** (pick your model) — it should respond.
4. After a job, **Open Console → Load activity & earnings** shows jobs + earnings.

## Troubleshooting
- **"Windows protected your PC"** → More info → Run anyway (unsigned beta).
- **Windows Defender quarantines sgl.exe** → allow it (Defender → Protection history →
  Allow), or add an exclusion for `%LOCALAPPDATA%\sgl-node`.
- **No GPU / crashes on load** → re-run `sgl.exe setup --cpu` and use a small model.
- **`llama-server` not found** → `sgl.exe setup` didn't finish; run it again and read the output.
- **Registered but "never serves"** → make sure this exact build's hash is allowlisted
  (it is for the beta) and your wallet actually has ≥ 50k SGL staked.
- **Firewall prompt** on first start → allow it (the node needs outbound to the grid).

Send any failure output to the team — copy the whole console window.
