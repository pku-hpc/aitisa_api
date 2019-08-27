Developer Guide {#dev_guide}
============================

AITISA_API is an open-source performance library for Standard APIs for AI operations. The library supports basic math operations beyond an unified data structure called Tensor and building blocks for neural networks. AITISA_API is also a baseline reference CPU version for High-performance Intelligent Compute Engine (HICE).

# Class
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
