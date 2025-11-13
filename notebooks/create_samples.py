import numpy as np

# Create two sample arrays
sample1 = np.random.rand(14512)  # Sample with size 14,512
sample2 = np.random.rand(14513)  # Sample with size 14,513

print(f"Sample 1 shape: {sample1.shape}")
print(f"Sample 2 shape: {sample2.shape}")

# Save the samples to files
np.save('sample_14512.npy', sample1)
np.save('sample_14513.npy', sample2) 