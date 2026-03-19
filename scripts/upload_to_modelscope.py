#!/usr/bin/env python3
"""
将训练好的模型上传到ModelScope
"""

import os
import argparse
from pathlib import Path


def upload_model(checkpoint_path: str, repo_name: str, token: str):
    from modelscope.hub.api import HubApi

    hub_api = HubApi()
    hub_api.login(token)

    model_dir = Path(checkpoint_path)

    if not model_dir.exists():
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    hub_api.push_model(model_id=repo_name, model_dir=str(model_dir))

    print(f"Successfully uploaded to: https://modelscope.cn/models/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to ModelScope")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/featurize/work/llm-review-sys-SFT/outputs/paper_review_qwen2.5_7b_qlora/checkpoint-500",
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="ModelScope repo name, format: <username>/<model_name>",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="ModelScope API token (or set MODELSCOPE_API_TOKEN env var)",
    )

    args = parser.parse_args()

    upload_model(checkpoint_path=args.checkpoint_path, repo_name=args.repo_name, token=args.token)


if __name__ == "__main__":
    main()
