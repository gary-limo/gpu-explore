import torch
import time

# Check for device availability
device_gpu = 'mps' if torch.backends.mps.is_available() else None
device_cpu = 'cpu'

# Define larger matrices (e.g., 1000x1000)
matrix_size = 9000
matrix_a = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
matrix_b = torch.randn(matrix_size, matrix_size, dtype=torch.float32)

# Function to perform matrix multiplication 1000 times and measure time
def matrix_mult_loop(a, b, device, iterations=100):
    a, b = a.to(device), b.to(device)
    
    # Warmup phase for GPU
    if device != 'cpu':
        print("Warming up GPU...")
        for _ in range(10):
            _ = torch.mm(a, b)
        torch.mps.synchronize()
        

    start = time.time()
    
    print(f"Starting {iterations} iterations on {device}...")
    
    for i in range(iterations):
        result = torch.mm(a, b)
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1} iterations...")
    
    if device != 'cpu':
        torch.mps.synchronize()
    end = time.time()
    
    total_time = end - start
    avg_time = total_time / iterations
    print(f"\nResults for {device}:")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Average time per multiplication: {avg_time:.6f} seconds")
    print(f"Multiplications per second: {1/avg_time:.2f}")
    
    return result.cpu()

# Menu-driven program
while True:
    print("\nSelect the device for computation:")
    print("1. CPU")
    if device_gpu:
        print("2. GPU (MPS)")
    print("3. Exit")

    choice = input("Enter your choice: ")
    if choice == "1":
        print("\nRunning on CPU...")
        matrix_mult_loop(matrix_a, matrix_b, device_cpu)
    elif choice == "2" and device_gpu:
        print("\nRunning on GPU (MPS)...")
        matrix_mult_loop(matrix_a, matrix_b, device_gpu)
    elif choice == "3":
        print("\nExiting program. Goodbye!")
        break
    else:
        print("\nInvalid choice. Please try again.")
