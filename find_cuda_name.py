import torch

# 1. Check for CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")
    # You would typically exit or use CPU for device assignments
    # device0 = torch.device("cpu")
else:
    # 2. Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Total GPUs detected: {num_gpus}")

    # Optional: Print the name of each GPU for reference
    print("\nAvailable GPU Devices:")
    for i in range(num_gpus):
        print(f"  cuda:{i}: {torch.cuda.get_device_name(i)}")

    # 3. Create device objects and map models

    # device0 is 'cuda:0', device1 is 'cuda:1', and so on.
    if num_gpus >= 1:
        device0 = torch.device("cuda:0")
        # Example: model0.to(device0)
        print(f"\nModel 0 will be assigned to {device0}")

    if num_gpus >= 2:
        device1 = torch.device("cuda:1")
        # Example: model1.to(device1)
        print(f"Model 1 will be assigned to {device1}")

    # You can now use these device variables to move your models
    # model0.to(device0)
    # model1.to(device1)