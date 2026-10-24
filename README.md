---
title: TACIT Beyond the Manual
emoji: 🏭
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# TACIT: Beyond the Manual

Team FRF 202A · Thapar Japan Hackathon 2026 · JETRO-PS-03

- `/` home: player card and the four stages
- `/master` the master's game (the Inspection Line)
- `/junior` the junior's Dojo (eye, ear, hand, rank)

Each browser gets its own game, so several people can play at once.

## Deploy to Render (free)

1. Put this folder in a new GitHub repository (see below).
2. On render.com: New → Blueprint → connect GitHub → pick the repository → Apply.
   `render.yaml` sets everything up: Docker, free plan, health check.
3. Wait about 5–8 minutes for the first build. Your link is `https://tacit-xxxx.onrender.com`.

Every push to GitHub redeploys automatically. On the free plan the app sleeps after 15 minutes
without visitors; the first visit after that takes about a minute.

### Putting the folder on GitHub (Terminal)

Create an empty repository on github.com called `tacit` (no README), then:

```
cd ~/Desktop/tacit_render
git init
git add .
git commit -m "TACIT"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/tacit.git
git push -u origin main
```

When git asks for a password, paste a GitHub token (github.com → Settings → Developer settings →
Personal access tokens → Generate, tick `repo`), not your account password.

## Deploy to Hugging Face Spaces (free)

1. Make a free account at huggingface.co, then Settings → Access Tokens → New token, type **Write**. Copy it.
2. In Terminal, in this folder:

```
python3 -m pip install --user huggingface_hub
python3 deploy_to_hf.py
```

3. Paste the token when asked. The script prints your link. The first build takes about 5 minutes.

To update later, run `python3 deploy_to_hf.py` again.

## Other hosts

The `Dockerfile` also works on Render, Railway or any Docker host (it listens on `$PORT`, default 7860).
Run locally without Docker: `pip install -r requirements.txt` then `uvicorn server:app --app-dir src --port 8000`.
