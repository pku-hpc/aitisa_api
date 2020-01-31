extern "C" {
#include "src/core/tensor.h"
}

DataType dtypes[10];
Device devices[2];
LayoutType layout_types[2];
namespace {
struct ENUMInitializer {
  ENUMInitializer(){
    ::dtypes[0] = kInt8;
    ::dtypes[1] = kUint8;
    ::dtypes[2] = kInt16;
    ::dtypes[3] = kUint16;
    ::dtypes[4] = kInt32;
    ::dtypes[5] = kUint32;
    ::dtypes[6] = kInt64;
    ::dtypes[7] = kUint64;
    ::dtypes[8] = kFloat;
    ::dtypes[9] = kDouble;

    ::layout_types[0] = LAYOUT_DENSE;
    ::layout_types[1] = LAYOUT_SPARSE;

    ::devices[0].type = DEVICE_CPU;
    ::devices[0].id = 0;
    ::devices[1].type = DEVICE_CUDA;
    ::devices[1].id = 0;
  }
  ~ENUMInitializer(){}
};
ENUMInitializer inits();
} // anonymous namespace


#define TRANSLATE_DATA_TYPE(INT_TYPE, AITISA_TYPE)      \
  DataType AITISA_TYPE = dtypes[INT_TYPE];

#define TRANSLATE_DEVICE_TYPE(INT_TYPE, AITISA_TYPE)    \
  Device AITISA_TYPE = devices[INT_TYPE];

#define TRANSLATE_LAYOUT_TYPE(INT_TYPE, AITISA_TYPE)    \
  LayoutType AITISA_TYPE = layout_types[INT_TYPE];

#define REGISTER_TENSOR(TENSOR_TYPE, DATA_TYPE, DEVICE_TYPE, LAYOUT_TYPE, CREATE_FUNC, RESOLVE_FUNC)    \
  class TensorBase {                                                                                    \
  public:                                                                                               \
    using UserTensor = TENSOR_TYPE;                                                                     \
    using UserDataType = DATA_TYPE;                                                                     \
    using UserDevice = DEVICE_TYPE;                                                                     \
    using UserLayoutType = LAYOUT_TYPE;                                                                 \
    static void create(int dtype, int device, int layout_type,                                          \
                       int64_t *dims, int64_t ndim, void *data, UserTensor *tensor){                    \
      CREATE_FUNC(dtype, device, layout_type, dims, ndim, data, tensor);                                \
    }                                                                                                   \
    static void resolve(UserTensor tensor, int *dtype, int64_t *ndim, int64_t *dims_ptr, void *data){   \
      RESOLVE_FUNC(tensor, dtype, ndim, dims_ptr, data);                                                \
    }                                                                                                   \
  };
