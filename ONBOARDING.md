# adversarial_rag: Onboarding

We edit code locally in VS Code (optionally with Claude Code) and run it on Colab through the official Colab extension. Google Drive carries the files.

## How it works

| Where | What | How it moves |
|---|---|---|
| **GitHub** | Shared code | `git push` / `git pull` |
| **Your Drive:** `My Drive/colab/adversarial_rag` | Your copy of the code, for your Colab | `./push.sh` (laptop → Drive, one-way) |
| **Shared Drive folder:** `My Drive/colab/adversarial_rag-shared` | `data/` and `outputs/` | Colab writes; `./pull.sh` downloads |

`git push` shares code with your partner. `./push.sh` only updates **your** Drive so **your** Colab can run it. Never edit code in Drive: each `./push.sh` overwrites it.

## One-time setup

**1. Clone the repo** (accept the GitHub invite first). Keep the folder name `adversarial_rag`, because the scripts use it to build Drive paths.

```bash
mkdir -p ~/projects && cd ~/projects
git clone <repo-url> && cd adversarial_rag
```

**2. Install rclone:** `sudo -v ; curl https://rclone.org/install.sh | sudo bash` (Mac: `brew install rclone`).

**3. Create your own Google client ID** (rclone's shared one is being retired). Follow https://rclone.org/drive/#making-your-own-client-id. To publish the app, Google needs a homepage and privacy-policy URL. A `<username>.github.io` GitHub Pages site with two short HTML pages works. Alternatively, leave the app in Testing and run `rclone config reconnect gdrive:` when your login expires (weekly).

**4. Configure rclone:** run `rclone config`, then:

- New remote named **`gdrive`**, storage type `drive`.
- Paste your client ID and secret. Scope: `drive`. Advanced config: `n`.
- Web browser auth: `y`. On WSL, paste the printed `127.0.0.1` URL into your Windows browser. If that fails, answer `n` and use the `rclone authorize` command it prints, running it on Windows.
- Shared Drive: `n`.

Then test it: `rclone lsd gdrive:` and `rclone mkdir gdrive:colab`.

> The end of `rclone config` prints your token. Don't screenshot or share it.

**5. Add the shared folder.** In drive.google.com → **Shared with me**, right-click `adversarial_rag-shared` → **Organize → Add shortcut**, and place it in **My Drive/colab**. Check it with `rclone lsd gdrive:colab/adversarial_rag-shared`.

**6. First push:** `./push.sh --dry-run`, then `./push.sh`.

**7. VS Code:** run `code .` and install **Jupyter** (Microsoft) and **Colab** (by **Google**, not "Colab Sync"). On WSL, click **Install in WSL** if an extension shows that button. Then run **Developer: Reload Window**.

**8. Connect:** open a notebook → **Select Kernel → Colab → Auto Connect** (or **New Colab Server**) → sign in.

## Setup cell (first cell of every notebook)

```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys, importlib
sys.dont_write_bytecode = True
PROJ   = "/content/drive/MyDrive/colab/adversarial_rag"
SHARED = "/content/drive/MyDrive/colab/adversarial_rag-shared"
os.chdir(PROJ); sys.path.insert(0, PROJ)

!pip install -q -r requirements.txt
```

## Daily workflow

1. **Start:** `git pull`, then `./push.sh`. Connect the kernel and run the setup cell.
2. **Edit:** change code, run `./push.sh`, then `importlib.reload(module)` in the notebook. (`%autoreload` is broken on Colab's runtime with `No module named 'imp'`, so reload manually.)
3. **Save results** to `f"{SHARED}/outputs/<your-name>/"`. Checkpoint long runs regularly.
4. **Finish:** `drive.flush_and_unmount()` in the notebook, `./pull.sh` locally, then commit and `git push`.

Tip: add `After editing any .py files, run ./push.sh.` to `CLAUDE.md` so Claude Code syncs for you.

## Pinned upstream RobustRAG code

The authors' Python source used by the experiments is committed in
`third_party/robustrag_upstream` at commit
`9bc35b2fa5fa7d1088383b2789ec19a512d316b1`. Notebooks use that local snapshot;
they do not clone the authors' GitHub repository in Colab.

The committed snapshot preserves the upstream Python source, with terminal
newlines normalized where absent. The shared setup helper checks its hashes,
copies it to `/content/robustrag-runner`, and applies the documented GPT-import
compatibility change only to that temporary copy. The large JSON datasets
remain under the shared Drive and are downloaded only when missing, with
checksum verification.

Changing the pinned commit is an experimental-methodology change. Update the
snapshot, hashes, provenance note, and experiment configuration together; do
not silently replace individual upstream files.

## Rules

- Code moves only through git. Commit small and often, and run `git pull` before you start.
- Each person owns their own notebooks (`notebooks/<name>_*.ipynb`). Shared logic goes in `.py` files. Clear notebook outputs before committing.
- Write outputs only to your own subfolder.
- Data stays out of git: upload it once to `SHARED/data`, and copy it to `/content` before training (`!cp -r {SHARED}/data /content/data`).
- No secrets in code. `.env` is gitignored and excluded from the push.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cell hangs, or `404 (Not Found)` | The runtime was recycled (idle ~90 min, max ~12 h on free). Run **Colab: Remove Server**, reconnect, and rerun the setup cell. |
| `FileNotFoundError` on `chdir` | Run `!ls /content/drive/MyDrive/colab`. Either you haven't pushed yet, or the folder name doesn't match. |
| Edit doesn't show up | Save the file and run `./push.sh`. Compare `rclone cat gdrive:colab/adversarial_rag/<file>` with `!cat {PROJ}/<file>`. If Colab shows the old text, run `drive.mount("/content/drive", force_remount=True)`. |
| Shared folder missing | The shortcut must be at `My Drive/colab/adversarial_rag-shared`. |
| rclone auth errors | Run `rclone config reconnect gdrive:`. |
| Colab terminal 404 on a working runtime | The terminal feature is experimental. Update the extension, or use `!` commands in cells. |
