# Build the image
docker compose build

# Run training on GPU 0
docker compose up train

# Quick test (5 epochs)
docker compose up train-quick

# Run benchmark for comparison
docker compose up benchmark

# Use specific GPU
CUDA_VISIBLE_DEVICES=1 docker compose up train

# Use multiple GPUs
docker compose up train-multi