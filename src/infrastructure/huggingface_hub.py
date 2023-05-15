from huggingface_hub import HfApi


def get_huggingface_hub_connection() -> HfApi:
    return HfApi()


def upload_results_file(api: HfApi):
    api.upload_file(
        path_or_fileobj="/path/to/local/folder/README.md",
        path_in_repo="README.md",
        repo_id="username/test-dataset",
        repo_type="dataset",
    )
