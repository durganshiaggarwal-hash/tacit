"""Put TACIT online as a free Hugging Face Space.
Usage:  python3 deploy_to_hf.py            (Space name: tacit)
        python3 deploy_to_hf.py my-name    (another Space name)"""
import os, sys, getpass
try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("First run:  python3 -m pip install --user huggingface_hub")

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "tacit"
token = os.environ.get("HF_TOKEN") or getpass.getpass("Paste your Hugging Face token (Write), then press Enter: ").strip()
api = HfApi(token=token)
user = api.whoami()["name"]
repo = f"{user}/{name}"
print(f"Creating the Space {repo} ...")
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
print("Uploading the app (about 50 MB, a few minutes) ...")
api.upload_folder(folder_path=HERE, repo_id=repo, repo_type="space", commit_message="Deploy TACIT",
                  ignore_patterns=["__pycache__/*", "**/__pycache__/*", "*.pyc", ".DS_Store", "**/.DS_Store", "deploy_to_hf.py"])
slug = f"{user}-{name}".lower().replace("_", "-").replace(".", "-")
print("\nDone. The first build takes about 5 minutes.")
print(f"  Build page : https://huggingface.co/spaces/{repo}")
print(f"  App link   : https://{slug}.hf.space")
