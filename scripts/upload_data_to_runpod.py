#!/usr/bin/env python3
"""
NeuroFusion-AD: Data Upload Helper
Uploads local ADNI and Bio-Hermes data to RunPod via SCP or RunPod CLI.

Usage:
    python scripts/upload_data_to_runpod.py --pod-ip <IP> --port <PORT> --key ~/.ssh/id_rsa
    python scripts/upload_data_to_runpod.py --list-files   # preview what will be uploaded

Get your pod IP and port from RunPod dashboard → Connect → SSH over exposed TCP
"""

import subprocess
import sys
import argparse
from pathlib import Path

# ── Configure these ───────────────────────────────────────────────────────────
RUNPOD_USER = "root"
REMOTE_BASE = "/workspace/neurofusion-ad/data/raw"

# Files to upload — edit paths if yours differ
ADNI_FILES = [
    "data/raw/adni/ADNIMERGE.csv",
    "data/raw/adni/UPENNBIOMK_MASTER.csv",   # or UPENNBIOMK9.csv
    "data/raw/adni/APOERES.csv",
    "data/raw/adni/DXSUM_PDXCONV_ADNIALL.csv",  # may be named differently
    "data/raw/adni/REGISTRY.csv",
]

BIOHERMES_FILES = [
    # Add all Bio-Hermes-001 CSV files here
    # Example: "data/raw/biohermes/biohermes001_plasma_biomarkers.csv",
]

def list_files():
    """Preview what would be uploaded."""
    print("ADNI files to upload:")
    for f in ADNI_FILES:
        p = Path(f)
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  ✅ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ MISSING: {f}")

    print("\nBio-Hermes-001 files to upload:")
    if not BIOHERMES_FILES:
        print("  ⚠️  No Bio-Hermes files configured — edit this script to add them")
    for f in BIOHERMES_FILES:
        p = Path(f)
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  ✅ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ MISSING: {f}")

def upload(pod_ip: str, port: int, key_path: str):
    """Upload all data files via SCP."""
    ssh_opts = [
        "-o", "StrictHostKeyChecking=no",
        "-P", str(port),
        "-i", key_path,
    ]

    def scp_file(local: str, remote_dir: str):
        local_p = Path(local)
        if not local_p.exists():
            print(f"  ⚠️  Skipping missing file: {local}")
            return False
        size_mb = local_p.stat().st_size / 1e6
        print(f"  Uploading {local} ({size_mb:.1f} MB)...")
        cmd = ["scp"] + ssh_opts + [local, f"{RUNPOD_USER}@{pod_ip}:{remote_dir}/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Done")
            return True
        else:
            print(f"  ❌ Failed: {result.stderr}")
            return False

    # Create remote directories first
    print("Creating remote directories...")
    mk_cmd = ["ssh"] + ["-o", "StrictHostKeyChecking=no", "-p", str(port), "-i", key_path,
                        f"{RUNPOD_USER}@{pod_ip}",
                        f"mkdir -p {REMOTE_BASE}/adni {REMOTE_BASE}/biohermes"]
    subprocess.run(mk_cmd)

    print("\nUploading ADNI files...")
    for f in ADNI_FILES:
        scp_file(f, f"{REMOTE_BASE}/adni")

    print("\nUploading Bio-Hermes-001 files...")
    for f in BIOHERMES_FILES:
        scp_file(f, f"{REMOTE_BASE}/biohermes")

    print("\n✅ Upload complete!")
    print(f"Verify on pod: ls -lh {REMOTE_BASE}/adni/ && ls -lh {REMOTE_BASE}/biohermes/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-files", action="store_true", help="Preview files to upload")
    parser.add_argument("--pod-ip", type=str, help="RunPod pod IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port (from RunPod dashboard)")
    parser.add_argument("--key", type=str, default="~/.ssh/id_rsa", help="SSH private key path")
    args = parser.parse_args()

    if args.list_files:
        list_files()
    elif args.pod_ip:
        upload(args.pod_ip, args.port, args.key)
    else:
        print("Usage:")
        print("  python scripts/upload_data_to_runpod.py --list-files")
        print("  python scripts/upload_data_to_runpod.py --pod-ip <IP> --port <PORT> --key ~/.ssh/id_rsa")
        print("\nAdd your Bio-Hermes file paths to BIOHERMES_FILES in this script first.")
