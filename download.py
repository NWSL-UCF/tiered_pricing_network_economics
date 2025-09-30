import kagglehub

# Download latest version
path = kagglehub.dataset_download("ittibydgoszcz/appraise-h2020-real-labelled-netflow-dataset")

print("Path to dataset files:", path)