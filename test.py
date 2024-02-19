# import numpy as np
import torch
import tensorflow as tf

# print(torch.cuda.is_available())
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available)
