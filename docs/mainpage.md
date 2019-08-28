Developer Guide {#dev_guide}
============================

AITISA_API is an open-source library aiming to provide a reference implementation for the standard APIs proposed by AITISA, which are widely used by different AI applications. Base on the unified data structures, this library will provide the basic math operations, the neural network operations, and the machine learning operations, etc. Besides, AITISA_API will also provide a testing framework to help other standard-compliant implementations to verify their correctness.

# Data Structures 
- @ref Allocator
- @ref DataType
- @ref Device
- @ref Layout
- @ref Shape
- @ref Tensor
- @ref _TensorImpl
- @ref _StorageImpl

# Basic Operations
- @ref aitisa_create
- @ref aitisa_destroy
- @ref aitisa_full
- @ref aitisa_tensor_data
- @ref aitisa_tensor_data_type
- @ref aitisa_tensor_device
- @ref aitisa_tensor_dim
- @ref aitisa_tensor_dims
- @ref aitisa_tensor_layout_type
- @ref aitisa_tensor_ndim
- @ref aitisa_tensor_set_item
- @ref aitisa_tensor_shape
- @ref aitisa_tensor_size
- @ref aitisa_tensor_storage
- @ref aitisa_get_stride
- @ref aitisa_get_all_strides
- @ref aitisa_coords_to_linidx
- @ref aitisa_linidx_to_coords
- @ref aitisa_coords_to_offset

# Math Operations
- @ref aitisa_add
- @ref aitisa_sub
- @ref aitisa_mul
- @ref aitisa_div
- @ref aitisa_matmul

# NN Operations
- @ref aitisa_conv
