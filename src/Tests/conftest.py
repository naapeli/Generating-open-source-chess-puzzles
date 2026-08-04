import os
# Disable PyTorch JIT compilation globally for all tests to avoid compilation dependencies/delays
os.environ["TORCH_COMPILE_DISABLE"] = "1"
