"""
AWS utilities for S3 storage and EC2 training.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError


class S3Manager:
    """
    Manages S3 operations for dataset storage and checkpoint saving.
    """

    def __init__(self, bucket_name: str, prefix: str = "interpremol"):
        """
        Args:
            bucket_name: S3 bucket name
            prefix: Prefix for all S3 keys (like a folder)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3 = boto3.client('s3')

    def _key(self, path: str) -> str:
        """Generate full S3 key with prefix."""
        return f"{self.prefix}/{path}".strip('/')

    def upload_file(self, local_path: str, s3_path: str) -> str:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path
            s3_path: Path within the bucket (prefix will be added)

        Returns:
            Full S3 URI
        """
        key = self._key(s3_path)
        self.s3.upload_file(local_path, self.bucket_name, key)
        return f"s3://{self.bucket_name}/{key}"

    def download_file(self, s3_path: str, local_path: str) -> str:
        """
        Download a file from S3.

        Args:
            s3_path: Path within the bucket (prefix will be added)
            local_path: Local destination path

        Returns:
            Local file path
        """
        key = self._key(s3_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket_name, key, local_path)
        return local_path

    def upload_directory(self, local_dir: str, s3_path: str) -> str:
        """
        Upload a directory to S3 (recursively).

        Args:
            local_dir: Local directory path
            s3_path: S3 destination path

        Returns:
            S3 URI of the directory
        """
        local_dir = Path(local_dir)
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                s3_file_path = f"{s3_path}/{relative_path}"
                self.upload_file(str(file_path), s3_file_path)

        return f"s3://{self.bucket_name}/{self._key(s3_path)}"

    def download_directory(self, s3_path: str, local_dir: str) -> str:
        """
        Download a directory from S3.

        Args:
            s3_path: S3 path (prefix will be added)
            local_dir: Local destination directory

        Returns:
            Local directory path
        """
        key_prefix = self._key(s3_path)
        paginator = self.s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=key_prefix):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                relative_path = s3_key[len(key_prefix):].lstrip('/')
                local_path = Path(local_dir) / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket_name, s3_key, str(local_path))

        return local_dir

    def list_files(self, s3_path: str = "") -> list:
        """List files in an S3 path."""
        key_prefix = self._key(s3_path)
        paginator = self.s3.get_paginator('list_objects_v2')

        files = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=key_prefix):
            for obj in page.get('Contents', []):
                files.append(obj['Key'])

        return files

    def file_exists(self, s3_path: str) -> bool:
        """Check if a file exists in S3."""
        key = self._key(s3_path)
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False


class CheckpointManager:
    """
    Manages model checkpoints with optional S3 backup.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        s3_manager: Optional[S3Manager] = None,
        s3_path: str = "checkpoints",
        keep_last_n: int = 3
    ):
        """
        Args:
            checkpoint_dir: Local directory for checkpoints
            s3_manager: Optional S3Manager for cloud backup
            s3_path: S3 path for checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.s3_manager = s3_manager
        self.s3_path = s3_path
        self.keep_last_n = keep_last_n

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        config: Dict[str, Any],
        is_best: bool = False,
        extra: Optional[Dict] = None
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            model: InterpreMol model
            optimizer: Optimizer state
            epoch: Current epoch
            val_loss: Validation loss
            config: Training config
            is_best: Whether this is the best model so far
            extra: Extra data to save

        Returns:
            Path to saved checkpoint
        """
        import torch

        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'config': config,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }
        if extra:
            checkpoint.update(extra)

        # Save epoch checkpoint
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            if self.s3_manager:
                self.s3_manager.upload_file(str(best_path), f"{self.s3_path}/best_model.pt")

        # Upload to S3
        if self.s3_manager:
            self.s3_manager.upload_file(str(filepath), f"{self.s3_path}/{filename}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return str(filepath)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for ckpt in checkpoints[self.keep_last_n:]:
            ckpt.unlink()

    def load_checkpoint(self, path: Optional[str] = None, load_best: bool = False) -> Dict:
        """
        Load a checkpoint.

        Args:
            path: Specific checkpoint path (local or S3)
            load_best: Load the best model instead

        Returns:
            Checkpoint dict
        """
        import torch

        if load_best:
            local_path = self.checkpoint_dir / "best_model.pt"
            if not local_path.exists() and self.s3_manager:
                self.s3_manager.download_file(
                    f"{self.s3_path}/best_model.pt",
                    str(local_path)
                )
        elif path:
            if path.startswith("s3://") and self.s3_manager:
                # Download from S3
                s3_key = path.replace(f"s3://{self.s3_manager.bucket_name}/", "")
                local_path = self.checkpoint_dir / Path(path).name
                self.s3_manager.download_file(s3_key, str(local_path))
            else:
                local_path = Path(path)
        else:
            # Load latest
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            local_path = checkpoints[0]

        return torch.load(local_path)

    def get_latest_epoch(self) -> int:
        """Get the epoch number of the latest checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not checkpoints:
            return 0

        # Extract epoch from filename
        name = checkpoints[0].stem
        epoch = int(name.split('_')[-1])
        return epoch

    def save_results(self, results: Dict, filename: str = "results.json"):
        """Save results to JSON (locally and to S3)."""
        local_path = self.checkpoint_dir / filename
        with open(local_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if self.s3_manager:
            self.s3_manager.upload_file(str(local_path), f"{self.s3_path}/{filename}")

        return str(local_path)


def create_s3_bucket(bucket_name: str, region: str = "us-east-1") -> bool:
    """
    Create an S3 bucket if it doesn't exist.

    Args:
        bucket_name: Name of the bucket
        region: AWS region

    Returns:
        True if created, False if already exists
    """
    s3 = boto3.client('s3', region_name=region)

    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"Created bucket: {bucket_name}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"Bucket already exists: {bucket_name}")
            return False
        raise


def get_instance_metadata() -> Dict:
    """
    Get EC2 instance metadata (when running on EC2).

    Returns:
        Dict with instance info or empty dict if not on EC2
    """
    import requests

    try:
        # IMDSv2 token
        token_response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=1
        )
        token = token_response.text

        headers = {"X-aws-ec2-metadata-token": token}

        instance_id = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers=headers, timeout=1
        ).text

        instance_type = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers=headers, timeout=1
        ).text

        return {
            'instance_id': instance_id,
            'instance_type': instance_type,
            'on_ec2': True
        }
    except:
        return {'on_ec2': False}
