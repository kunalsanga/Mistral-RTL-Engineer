"""
download_model.py — Robust model downloader with retry + resume support.
Run this ONCE before train.py. After it completes, train.py loads from cache instantly.
"""
import os, time, sys
from huggingface_hub import snapshot_download

MODEL_ID   = "unsloth/mistral-7b-v0.3-bnb-4bit"
MAX_TRIES  = 10
RETRY_WAIT = 15  # seconds between retries

print(f"Downloading: {MODEL_ID}")
print("This is ~4GB. Will auto-retry on network errors.\n")

for attempt in range(1, MAX_TRIES + 1):
    try:
        print(f"[Attempt {attempt}/{MAX_TRIES}] Downloading...")
        path = snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*", "rust_*"],
            resume_download=True,     # resumes partial downloads
            max_workers=1,            # single thread = more stable on slow connections
        )
        print(f"\n✅ Download complete! Cached at:\n{path}")
        print("\nYou can now run: python train.py")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"[Attempt {attempt}] Error: {e}")
        if attempt < MAX_TRIES:
            print(f"Retrying in {RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)
        else:
            print("All attempts failed. Check your internet connection.")
            sys.exit(1)
