"""
Upload trained models to Google Cloud Storage.
"""
import argparse
import sys
from pathlib import Path
from google.cloud import storage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.auth.gcp_auth import authenticate_gcp
from config.settings import GCS_BUCKET, MODELS_DIR, PREPROCESSORS_DIR


def upload_directory_to_gcs(
    bucket_name: str,
    source_dir: Path,
    destination_prefix: str
):
    """
    Upload directory contents to GCS.
    
    Args:
        bucket_name: GCS bucket name
        source_dir: Local directory to upload
        destination_prefix: GCS prefix/folder
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    uploaded_files = []
    
    for local_file in source_dir.rglob("*"):
        if local_file.is_file():
            # Compute relative path
            relative_path = local_file.relative_to(source_dir)
            blob_name = f"{destination_prefix}/{relative_path}"
            
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_file))
            
            uploaded_files.append(blob_name)
            print(f"  ✅ {blob_name}")
    
    return uploaded_files


def main():
    parser = argparse.ArgumentParser(description="Upload models to GCS")
    
    parser.add_argument(
        "--bucket",
        type=str,
        default=GCS_BUCKET,
        help="GCS bucket name"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="models",
        help="GCS prefix/folder"
    )
    
    parser.add_argument(
        "--use-colab",
        action="store_true",
        help="Use Colab authentication"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Authenticate
    authenticate_gcp(use_colab=args.use_colab)
    
    print(f"\n📤 Uploading models to gs://{args.bucket}/{args.prefix}/")
    print("=" * 60)
    
    # Upload models
    print("\nUploading model artifacts...")
    model_files = upload_directory_to_gcs(
        args.bucket,
        MODELS_DIR,
        f"{args.prefix}/models"
    )
    
    # Upload preprocessors
    print("\nUploading preprocessor artifacts...")
    prep_files = upload_directory_to_gcs(
        args.bucket,
        PREPROCESSORS_DIR,
        f"{args.prefix}/preprocessors"
    )
    
    print("\n" + "=" * 60)
    print(f"✅ Upload complete!")
    print(f"Total files uploaded: {len(model_files) + len(prep_files)}")
    print(f"GCS location: gs://{args.bucket}/{args.prefix}/")
    print("=" * 60)


if __name__ == "__main__":
    main()