
# run it with CUBLAS_WORKSPACE_CONFIG=:4096:8 python nondeterminism_example.py to ensure CUBLAS determinism

import torch
import numpy as np



# make matmul comparable (disable TF32 fast path)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# determinism guard (PyTorch side). For matmul on CUDA you ALSO need the env var above.
torch.use_deterministic_algorithms(True, warn_only=False)


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

x = torch.randn(2500, 2500)

def matrix_polynomial(x):
    y = x + 0.5 * x @ x + 3.0 * x @ x @ x + x @ x @ x @ x
    return (y).sum().item()

print("Torch CPU:", matrix_polynomial(x), sep='\t')
print("Torch GPU:", matrix_polynomial(x.cuda()), sep='\t')

print("Numpy: ", matrix_polynomial(x.numpy()), sep='\t')
print("Numpy built-in: ", (x.numpy() +
                           0.5 * np.linalg.matrix_power(x.numpy(), 2) +
                           3.0 * np.linalg.matrix_power(x.numpy(), 3) +
                           np.linalg.matrix_power(x.numpy(), 4)
                           ).sum(), sep='\t')
