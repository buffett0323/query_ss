import kagglehub

# Download latest version
path = kagglehub.dataset_download("imsparsh/musicnet-dataset")

print("Path to dataset files:", path)
