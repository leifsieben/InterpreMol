#!/usr/bin/env python3
"""
AWS Setup Script for InterpreMol

This script helps set up AWS infrastructure for pretraining:
1. Create S3 bucket
2. Upload dataset to S3
3. Launch EC2 instance with GPU
4. Set up the training environment

Prerequisites:
- AWS CLI configured with credentials (aws configure)
- boto3 installed (pip install boto3)

Usage:
    python setup_aws.py --create-bucket --bucket-name interpremol-data
    python setup_aws.py --upload-dataset --bucket-name interpremol-data
    python setup_aws.py --launch-instance --bucket-name interpremol-data
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


# EC2 Configuration
EC2_CONFIG = {
    # Recommended instances for GPU training
    "instances": {
        "hyperopt": {
            "type": "g5.2xlarge",  # 1x A10G (24GB), 8 vCPU, 32GB RAM - $1.21/hr
            "description": "Good for hyperparameter optimization"
        },
        "pretrain_small": {
            "type": "g5.4xlarge",  # 1x A10G (24GB), 16 vCPU, 64GB RAM - $2.42/hr
            "description": "More CPU/RAM for data loading"
        },
        "pretrain_large": {
            "type": "p3.2xlarge",  # 1x V100 (16GB), 8 vCPU, 61GB RAM - $3.06/hr
            "description": "V100 GPU, good for large models"
        },
        "pretrain_multi": {
            "type": "g5.12xlarge",  # 4x A10G (24GB each), 48 vCPU, 192GB RAM - $7.26/hr
            "description": "Multi-GPU training"
        }
    },
    # Deep Learning AMI (PyTorch)
    "ami": {
        "us-east-1": "ami-0c7217cdde317cfec",  # Ubuntu 22.04
        "us-west-2": "ami-0efcece6bed30fd98",
    },
    "key_name": "interpremol-key",
    "security_group": "interpremol-sg",
    "volume_size": 200,  # GB
}

# Startup script for EC2 instance
STARTUP_SCRIPT = """#!/bin/bash
set -e

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git awscli

# Create working directory
mkdir -p /home/ubuntu/interpremol
cd /home/ubuntu/interpremol

# Clone repository (or sync from S3)
aws s3 sync s3://{bucket}/code/ /home/ubuntu/interpremol/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Download dataset from S3
mkdir -p datasets
aws s3 cp s3://{bucket}/datasets/all_datasets_fused_standardized.parquet datasets/

# Create checkpoints directory
mkdir -p checkpoints

echo "Setup complete! Run: source venv/bin/activate && cd /home/ubuntu/interpremol/model"
"""


def create_bucket(bucket_name: str, region: str = "us-east-1"):
    """Create S3 bucket."""
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

        # Enable versioning
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        print("Enabled versioning")

    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"Bucket already exists: {bucket_name}")
        else:
            raise


def upload_dataset(bucket_name: str, dataset_path: str):
    """Upload dataset to S3."""
    s3 = boto3.client('s3')

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    s3_key = f"datasets/{dataset_path.name}"
    file_size = dataset_path.stat().st_size / (1024 * 1024)  # MB

    print(f"Uploading {dataset_path.name} ({file_size:.1f} MB)...")

    # Use multipart upload for large files
    from boto3.s3.transfer import TransferConfig
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100MB
        max_concurrency=10
    )

    s3.upload_file(
        str(dataset_path),
        bucket_name,
        s3_key,
        Config=config,
        Callback=lambda bytes_transferred: print(f"  {bytes_transferred / (1024*1024):.1f} MB", end='\r')
    )
    print(f"\nUploaded to s3://{bucket_name}/{s3_key}")


def upload_code(bucket_name: str, code_dir: str = "."):
    """Upload code to S3."""
    s3 = boto3.client('s3')
    code_dir = Path(code_dir)

    files_to_upload = [
        "model/*.py",
        "requirements.txt",
        "CLAUDE.md",
    ]

    import glob
    for pattern in files_to_upload:
        for filepath in glob.glob(str(code_dir / pattern)):
            filepath = Path(filepath)
            s3_key = f"code/{filepath.relative_to(code_dir)}"
            s3.upload_file(str(filepath), bucket_name, s3_key)
            print(f"Uploaded {filepath} -> s3://{bucket_name}/{s3_key}")


def create_key_pair(key_name: str, region: str = "us-east-1"):
    """Create EC2 key pair and save to file."""
    ec2 = boto3.client('ec2', region_name=region)

    try:
        response = ec2.create_key_pair(KeyName=key_name)
        key_material = response['KeyMaterial']

        key_file = Path.home() / f".ssh/{key_name}.pem"
        with open(key_file, 'w') as f:
            f.write(key_material)
        os.chmod(key_file, 0o400)

        print(f"Created key pair: {key_name}")
        print(f"Saved to: {key_file}")
        return str(key_file)

    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidKeyPair.Duplicate':
            print(f"Key pair already exists: {key_name}")
            return str(Path.home() / f".ssh/{key_name}.pem")
        raise


def create_security_group(sg_name: str, region: str = "us-east-1"):
    """Create security group for SSH access."""
    ec2 = boto3.client('ec2', region_name=region)

    try:
        # Get default VPC
        vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
        vpc_id = vpcs['Vpcs'][0]['VpcId']

        # Create security group
        response = ec2.create_security_group(
            GroupName=sg_name,
            Description="InterpreMol training instance",
            VpcId=vpc_id
        )
        sg_id = response['GroupId']

        # Allow SSH
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                'IpProtocol': 'tcp',
                'FromPort': 22,
                'ToPort': 22,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH'}]
            }]
        )

        print(f"Created security group: {sg_id}")
        return sg_id

    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidGroup.Duplicate':
            # Get existing group ID
            groups = ec2.describe_security_groups(GroupNames=[sg_name])
            sg_id = groups['SecurityGroups'][0]['GroupId']
            print(f"Security group already exists: {sg_id}")
            return sg_id
        raise


def create_iam_role(role_name: str = "interpremol-ec2-role"):
    """Create IAM role for EC2 to access S3."""
    iam = boto3.client('iam')

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for InterpreMol EC2 instances"
        )
        print(f"Created IAM role: {role_name}")

        # Attach S3 access policy
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
        )
        print("Attached S3 access policy")

        # Create instance profile
        iam.create_instance_profile(InstanceProfileName=role_name)
        iam.add_role_to_instance_profile(
            InstanceProfileName=role_name,
            RoleName=role_name
        )
        print(f"Created instance profile: {role_name}")

        # Wait for profile to be ready
        time.sleep(10)

    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f"IAM role already exists: {role_name}")
        else:
            raise

    return role_name


def launch_instance(
    bucket_name: str,
    instance_type: str = "g5.2xlarge",
    region: str = "us-east-1"
):
    """Launch EC2 instance for training."""
    ec2 = boto3.client('ec2', region_name=region)

    # Ensure prerequisites
    key_file = create_key_pair(EC2_CONFIG["key_name"], region)
    sg_id = create_security_group(EC2_CONFIG["security_group"], region)
    iam_role = create_iam_role()

    # Get AMI for region
    ami_id = EC2_CONFIG["ami"].get(region)
    if not ami_id:
        # Find latest Deep Learning AMI
        response = ec2.describe_images(
            Owners=['amazon'],
            Filters=[
                {'Name': 'name', 'Values': ['Deep Learning AMI GPU PyTorch*Ubuntu*']},
                {'Name': 'state', 'Values': ['available']}
            ]
        )
        if response['Images']:
            ami_id = sorted(response['Images'], key=lambda x: x['CreationDate'], reverse=True)[0]['ImageId']
        else:
            # Fallback to Ubuntu 22.04
            ami_id = "ami-0c7217cdde317cfec"

    print(f"Using AMI: {ami_id}")

    # Prepare startup script
    user_data = STARTUP_SCRIPT.format(bucket=bucket_name)

    # Launch instance
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=EC2_CONFIG["key_name"],
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        IamInstanceProfile={'Name': iam_role},
        BlockDeviceMappings=[{
            'DeviceName': '/dev/sda1',
            'Ebs': {
                'VolumeSize': EC2_CONFIG["volume_size"],
                'VolumeType': 'gp3',
                'DeleteOnTermination': True
            }
        }],
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [
                {'Key': 'Name', 'Value': 'interpremol-training'},
                {'Key': 'Project', 'Value': 'interpremol'}
            ]
        }]
    )

    instance_id = response['Instances'][0]['InstanceId']
    print(f"Launched instance: {instance_id}")

    # Wait for instance to be running
    print("Waiting for instance to start...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])

    # Get public IP
    instance = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = instance['Reservations'][0]['Instances'][0].get('PublicIpAddress')

    print(f"\nInstance is running!")
    print(f"Instance ID: {instance_id}")
    print(f"Public IP: {public_ip}")
    print(f"\nConnect with:")
    print(f"  ssh -i {key_file} ubuntu@{public_ip}")
    print(f"\nNote: Wait a few minutes for setup script to complete.")
    print(f"Check progress with: tail -f /var/log/cloud-init-output.log")

    return instance_id, public_ip


def list_instances(region: str = "us-east-1"):
    """List running InterpreMol instances."""
    ec2 = boto3.client('ec2', region_name=region)

    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Project', 'Values': ['interpremol']},
            {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
        ]
    )

    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append({
                'id': instance['InstanceId'],
                'type': instance['InstanceType'],
                'ip': instance.get('PublicIpAddress'),
                'state': instance['State']['Name']
            })

    if instances:
        print("Running instances:")
        for inst in instances:
            print(f"  {inst['id']} ({inst['type']}) - {inst['ip']} - {inst['state']}")
    else:
        print("No running instances found")

    return instances


def stop_instance(instance_id: str, region: str = "us-east-1"):
    """Stop an EC2 instance."""
    ec2 = boto3.client('ec2', region_name=region)
    ec2.stop_instances(InstanceIds=[instance_id])
    print(f"Stopping instance: {instance_id}")


def terminate_instance(instance_id: str, region: str = "us-east-1"):
    """Terminate an EC2 instance."""
    ec2 = boto3.client('ec2', region_name=region)
    ec2.terminate_instances(InstanceIds=[instance_id])
    print(f"Terminating instance: {instance_id}")


def main():
    parser = argparse.ArgumentParser(description="AWS Setup for InterpreMol")
    parser.add_argument("--region", default="us-east-1", help="AWS region")

    # Actions
    parser.add_argument("--create-bucket", action="store_true", help="Create S3 bucket")
    parser.add_argument("--upload-dataset", action="store_true", help="Upload dataset to S3")
    parser.add_argument("--upload-code", action="store_true", help="Upload code to S3")
    parser.add_argument("--launch-instance", action="store_true", help="Launch EC2 instance")
    parser.add_argument("--list-instances", action="store_true", help="List running instances")
    parser.add_argument("--stop-instance", type=str, help="Stop instance by ID")
    parser.add_argument("--terminate-instance", type=str, help="Terminate instance by ID")

    # Options
    parser.add_argument("--bucket-name", type=str, help="S3 bucket name")
    parser.add_argument("--dataset-path", type=str,
                       default="datasets/all_datasets_fused_standardized.parquet",
                       help="Path to dataset file")
    parser.add_argument("--instance-type", type=str, default="g5.2xlarge",
                       choices=list(EC2_CONFIG["instances"].keys()) + ["g5.2xlarge", "g5.4xlarge", "p3.2xlarge"],
                       help="EC2 instance type")

    args = parser.parse_args()

    # Map instance type aliases
    if args.instance_type in EC2_CONFIG["instances"]:
        args.instance_type = EC2_CONFIG["instances"][args.instance_type]["type"]

    if args.create_bucket:
        if not args.bucket_name:
            print("Error: --bucket-name required")
            sys.exit(1)
        create_bucket(args.bucket_name, args.region)

    if args.upload_dataset:
        if not args.bucket_name:
            print("Error: --bucket-name required")
            sys.exit(1)
        upload_dataset(args.bucket_name, args.dataset_path)

    if args.upload_code:
        if not args.bucket_name:
            print("Error: --bucket-name required")
            sys.exit(1)
        upload_code(args.bucket_name, str(Path(__file__).parent.parent))

    if args.launch_instance:
        if not args.bucket_name:
            print("Error: --bucket-name required")
            sys.exit(1)
        launch_instance(args.bucket_name, args.instance_type, args.region)

    if args.list_instances:
        list_instances(args.region)

    if args.stop_instance:
        stop_instance(args.stop_instance, args.region)

    if args.terminate_instance:
        terminate_instance(args.terminate_instance, args.region)

    if not any([args.create_bucket, args.upload_dataset, args.upload_code,
                args.launch_instance, args.list_instances,
                args.stop_instance, args.terminate_instance]):
        parser.print_help()
        print("\n\nRecommended instance types:")
        for name, info in EC2_CONFIG["instances"].items():
            print(f"  {name}: {info['type']} - {info['description']}")


if __name__ == "__main__":
    main()
