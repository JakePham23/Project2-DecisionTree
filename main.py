import kagglehub

# Download latest version
path = kagglehub.dataset_download("./data/heart_disease.csv")

print("Path to dataset files:", path)