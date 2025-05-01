from utils.check import hip_check, compare  
import pydra

from hip import hip, hipblas
import numpy as np
import torch

def test_pytorch_matmul(config: pydra.Config, M: int, N: int, K: int, A_h: np.ndarray, B_h: np.ndarray, C_h: np.ndarray, alpha: float, beta: float, C_expected: np.ndarray):
    """
    Test the performance of PyTorch matrix multiplication on AMD GPU with ROCm
    A_h, B_h, C_h are numpy arrays
    """
    # For AMD GPUs with ROCm, we still use "cuda" as the device name in PyTorch
    # Check if ROCm-enabled PyTorch is available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Make sure PyTorch is built with ROCm support")
    
    # device = torch.device("cpu")
    # device = torch.device("cuda")
    
    # Print device info for debugging
    device = torch.device("cuda:0")
    # print(f"Using PyTorch device: {device}")
    # print(f"Device name: {torch.cuda.get_device_name(0)}")

    # device = 
    # Convert numpy arrays to PyTorch tensors
    A_tensor = torch.from_numpy(A_h).to(device)
    B_tensor = torch.from_numpy(B_h).to(device)
    C_tensor = torch.from_numpy(C_h).to(device)

    # import pdb; pdb.set_trace()
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Record start event
    start_event.record()
    
    # import pdb; pdb.set_trace()

    # Perform matrix multiplication
    C_result = alpha * torch.matmul(A_tensor, B_tensor) + beta * C_tensor
    
    # Record end event and synchronize
    end_event.record()
    torch.cuda.synchronize()
    
    # Get elapsed time in milliseconds
    elapsed_time = start_event.elapsed_time(end_event)

    # Copy result back to CPU for validation
    C_result_cpu = C_result.cpu().numpy()
    compare(C_result_cpu, C_expected, config.debug)

    return elapsed_time

