import tensorflow as tf
import time

# Create a random tensor
a = tf.random.normal([10000, 10000])

# Measure time for matrix multiplication on GPU
start_time = time.time()
with tf.device('/GPU:0'):
    b = tf.matmul(a, a)
gpu_time = time.time() - start_time

# Measure time for matrix multiplication on CPU
start_time = time.time()
with tf.device('/CPU:0'):
    c = tf.matmul(a, a)
cpu_time = time.time() - start_time

print(f"GPU computation time: {gpu_time:.4f} seconds")
print(f"CPU computation time: {cpu_time:.4f} seconds")
