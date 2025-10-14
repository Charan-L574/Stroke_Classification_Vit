import torch

def check_gpu():
    """
    Checks for PyTorch GPU availability and prints the status.
    """
    print("--- GPU Diagnosis ---")
    try:
        if torch.cuda.is_available():
            print("✅ Success! PyTorch can see your GPU.")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version PyTorch was built with: {torch.version.cuda}")
            print("\nYour environment is set up correctly for GPU training.")
        else:
            print("❌ Failure: PyTorch cannot see your GPU.")
            print("\nThis is likely due to one of the following reasons:")
            print("1. You have a CPU-only version of PyTorch installed.")
            print("2. Your NVIDIA drivers are not installed correctly or are outdated.")
            print("3. The CUDA toolkit version is incompatible with your PyTorch version.")
            print("\nTo fix this, you may need to reinstall PyTorch with CUDA support.")
            print("Visit https://pytorch.org/get-started/locally/ for the correct installation command.")

    except Exception as e:
        print(f"An error occurred during GPU check: {e}")
        print("This could indicate a problem with your NVIDIA driver installation.")
        
    print("---------------------")

if __name__ == '__main__':
    check_gpu()
