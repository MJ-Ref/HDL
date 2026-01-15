#!/usr/bin/env python3
"""
Upload M2 training data to Modal volume.

Usage:
    modal run modal/upload_data.py --data-path data/m2_train.jsonl
"""

import modal

app = modal.App("m2-data-upload")

# Create a persistent volume for training data
volume = modal.Volume.from_name("m2-training-data", create_if_missing=True)


@app.function(volumes={"/data": volume})
def save_to_volume(content: bytes, remote_name: str):
    """Save content to the Modal volume."""
    from pathlib import Path

    remote_path = Path("/data") / remote_name
    remote_path.parent.mkdir(parents=True, exist_ok=True)

    # Write content to volume
    with open(remote_path, 'wb') as f:
        f.write(content)

    # Verify
    size = remote_path.stat().st_size
    print(f"Uploaded: {remote_name} ({size:,} bytes)")

    # Count lines (samples)
    with open(remote_path) as f:
        n_lines = sum(1 for _ in f)
    print(f"Samples: {n_lines}")

    volume.commit()
    return {"path": str(remote_path), "size": size, "samples": n_lines}


@app.local_entrypoint()
def main(data_path: str = "data/m2_train.jsonl"):
    """Upload training data to Modal volume."""
    from pathlib import Path

    local_file = Path(data_path)
    if not local_file.exists():
        print(f"Error: File not found: {data_path}")
        return

    print(f"Reading local file: {local_file}")
    content = local_file.read_bytes()
    print(f"File size: {len(content):,} bytes")

    remote_name = local_file.name
    print(f"Uploading to Modal volume as: {remote_name}")

    result = save_to_volume.remote(content, remote_name)

    print(f"\nUpload complete!")
    print(f"Remote path: {result['path']}")
    print(f"Size: {result['size']:,} bytes")
    print(f"Samples: {result['samples']}")
