from model import CliffordNet

# Execution flow (high-level):
# 1) `train.py` calls one of the constructors in this file (e.g. `cliffordnet_12_2`).
# 2) Each constructor returns a `CliffordNet` instance defined in `model.py`.
# 3) `CliffordNet` defines the model architecture and uses utilities from `utils.py`.

# Step 1: Helper utilities used to construct CliffordNet variants.
# These helpers define the shift patterns used in geometric interactions.

def gen_shifts(n):
    """Generate a list of bit-shift values (powers of two).

    Args:
        n: Number of shift values to generate.

    Returns:
        List of shifts [1, 2, 4, ..., 2^(n-1)].

    Note: These shifts determine the receptive field and interaction patterns.
    """
    return [1 << i for i in range(n)]

def gen_shifts_fibonacci(n):
    """Generate a Fibonacci-like sequence of shift values.

    This alternative shift pattern can be used as an experimental strategy
    for the Clifford interaction mechanism.
    """
    a, b = 1, 2
    for _ in range(n):
        yield a
        a, b = b, a + b
        
def cliffordnet_12_2(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    """Create a CliffordNet-12-2 configuration.

    This is a lightweight "nano" version with 12 layers and shift set [1,2].
    Use this for quick experiments or when compute is limited.
    """
    # Nano: shifts=[1, 2]
    shifts = gen_shifts(2)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=12,
        drop_path_rate=0.3
    )    

def cliffordnet_12_3(num_classes=100, patch_size=1, embed_dim=160, enable_cuda=False):
    """Create a CliffordNet-12-3 configuration.

    This is a lightweight variant with 12 layers and shift set [1,2,4].
    """
    # Nano: shifts=[1, 2, 4]
    shifts = gen_shifts(3)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=12,
        drop_path_rate=0.3 
    )    

def cliffordnet_12_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    """Create a CliffordNet-12-5 configuration.

    This is a lite version with 12 layers and larger shift set [1,2,4,8,16].
    """
    # Lite: shifts=[1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=12,
        drop_path_rate=0.3 
    )    

def cliffordnet_18_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    """Create a CliffordNet-18-5 configuration.

    A larger model with 18 layers and shift set [1,2,4,8,16].
    """
    # Lite: shifts=[1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=18,
        drop_path_rate=0.3 
    )  

def cliffordnet_32_3(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    """Create a CliffordNet-32-3 configuration.

    A small model with 32 layers and shift set [1,2,4].
    """
    # Small: 32 layers
    # Shifts: [1, 2, 4]
    shifts = gen_shifts(3)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=32,
        drop_path_rate=0.3 
    )     

def cliffordnet_32_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    """Create a CliffordNet-32-5 configuration.

    A small model with 32 layers and shift set [1,2,4,8,16], using `cli_mode='inner'`.
    """
    # Small: 32 layers
    # Shifts: [1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='inner', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=32,
        drop_path_rate=0.3 
    )     
 
def cliffordnet_64_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    """Create a CliffordNet-64-5 configuration.

    A deeper model with 64 layers and shift set [1,2,4,8,16].
    """
    # Deep: 64 layers
    # Shifts: [1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='inner', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=64,
        drop_path_rate=0.4 
    )     
