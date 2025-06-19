from google.cloud import storage
import zipfile

def download_model(bucket_name: str, model_name: str, local_path: str):
    client = storage.Client()  # wykorzystuje GOOGLE_APPLICATION_CREDENTIALS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)
    blob.download_to_filename(local_path)
    print(f"Pobrano model: {model_name} do {local_path}")

def download_and_extract_model(bucket_name, model_name, local_path):
    zip_path = local_path + ".zip"
    download_model(bucket_name, model_name, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_path)
    print(f"Model rozpakowany do {local_path}")

