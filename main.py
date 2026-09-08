def main():
    print("Hello from model-training!")


if __name__ == "__main__":
    main()
    import torch

    print("Torch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

# uv run python scripts/train_dinov2_arcface.py --batch-size 8 --grad-accum 4 --epochs 60
