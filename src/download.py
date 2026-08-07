import kagglehub

# Download latest version
path = kagglehub.dataset_download("subhajournal/drug-recommendations")

print("Path to dataset files:", path)