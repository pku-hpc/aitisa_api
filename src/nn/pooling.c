#include "src/core/utils.h"
#include "src/core/allocator.h"
#include "src/basic/index_utils.h"
#include "pooling.h"
#include <stdio.h>
#include <stdlib.h>

#define max_pooling_1d_kernel(typename)                                             \
    typename *turned_in_data = (typename *)in_data;                                 \
    typename *turned_out_data = (typename *)out_data;                               \
    for(int64_t channel=0; channel<nchannels_within_batch; channel++){              \
        int64_t out_offset = channel * out_feature_size;                            \
        int64_t in_offset = channel * in_feature_size;                              \
        int index_ktail = ksize[0] - 1 - left_padding;                              \
        for(int64_t out_feature=0; out_feature<out_feature_size; out_feature++){    \
            typename max;                                                           \
            if(index_ktail < in_feature_size){                                      \
                max = turned_in_data[in_offset + index_ktail];                      \
            }else{                                                                  \
                max = 0;                                                            \
            }                                                                       \
            int index_khead = index_ktail - ksize[0];                               \
            for(int i=index_ktail-1; i>index_khead; i--){                           \
                /* search for the max value within a window */                      \
                if(i < 0){                                                          \
                    /* deal with left padding */                                    \
                    if(max < 0){                                                    \
                        max = 0;                                                    \
                    }                                                               \
                    break;                                                          \
                }                                                                   \
                if(i >= in_feature_size){                                           \
                    /* deal with right padding */                                   \
                    continue;                                                       \
                }                                                                   \
                if(turned_in_data[in_offset+i] > max){                              \
                    max = turned_in_data[in_offset+i];                              \
                }                                                                   \
            }                                                                       \
            turned_out_data[out_offset+out_feature] = max;                          \
            index_ktail += stride[0];                                               \
        }                                                                           \
    }

static Status max_pooling_1d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             Tensor *output){
    int64_t *in_dims = aitisa_tensor_dims(input);
    int64_t *out_dims = aitisa_tensor_dims(*output);
    void* in_data = aitisa_tensor_data(input);
    int64_t in_feature_size = in_dims[2];
    void* out_data = aitisa_tensor_data(*output);
    int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
    int64_t out_feature_size = out_dims[2];
    int left_padding = padding[0] / 2;
    //int right_padding = padding[0] / 2 + (padding[0] % 2);

    switch(aitisa_tensor_data_type(input).code){
        case TYPE_INT8: {
            max_pooling_1d_kernel(int8_t);
            break;
        }
        case TYPE_UINT8: {
            max_pooling_1d_kernel(uint8_t);
            break;
        }
        case TYPE_INT16: {
            max_pooling_1d_kernel(int16_t);
            break;
        }
        case TYPE_UINT16: {
            max_pooling_1d_kernel(uint16_t);
            break;
        }
        case TYPE_INT32: {
            max_pooling_1d_kernel(int32_t);
            break;
        }
        case TYPE_UINT32: {
            max_pooling_1d_kernel(uint8_t);
            break;
        }
        case TYPE_INT64: {
            max_pooling_1d_kernel(int64_t);
            break;
        }
        case TYPE_UINT64: {
            max_pooling_1d_kernel(uint64_t);
            break;
        }
        case TYPE_FLOAT: {
            max_pooling_1d_kernel(float);
            break;
        }
        case TYPE_DOUBLE: {
            max_pooling_1d_kernel(double);
            break;
        }
        default:
            return STATUS_NOT_SUPPORTED;
    }

    return STATUS_SUCCESS;
}

#define avg_pooling_1d_kernel(typename)                                             \
    typename *turned_in_data = (typename *)in_data;                                 \
    typename *turned_out_data = (typename *)out_data;                               \
    for(int64_t channel=0; channel<nchannels_within_batch; channel++){              \
        int64_t out_offset = channel * out_feature_size;                            \
        int64_t in_offset = channel * in_feature_size;                              \
        int index_ktail = ksize[0] - 1 - left_padding;                              \
        for(int64_t out_feature=0; out_feature<out_feature_size; out_feature++){    \
            typename total = 0;                                                     \
            int index_khead = index_ktail - ksize[0];                               \
            for(int i=index_ktail; i>index_khead; i--){                             \
                /* search for the max value within a window */                      \
                if(i < 0){                                                          \
                    /* deal with left padding */                                    \
                    break;                                                          \
                }                                                                   \
                if(i >= in_feature_size){                                           \
                    /* deal with right padding */                                   \
                    continue;                                                       \
                }                                                                   \
                total += turned_in_data[in_offset+i];                               \
            }                                                                       \
            turned_out_data[out_offset+out_feature] = total / (typename)(ksize[0]); \
            index_ktail += stride[0];                                               \
        }                                                                           \
    }

static Status avg_pooling_1d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             Tensor *output){
    int64_t *in_dims = aitisa_tensor_dims(input);
    int64_t *out_dims = aitisa_tensor_dims(*output);
    void* in_data = aitisa_tensor_data(input);
    int64_t in_feature_size = in_dims[2];
    void* out_data = aitisa_tensor_data(*output);
    int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
    int64_t out_feature_size = out_dims[2];
    int left_padding = padding[0] / 2;
    //int right_padding = padding[0] / 2 + (padding[0] % 2);

    switch(aitisa_tensor_data_type(input).code){
        case TYPE_INT8: {
            avg_pooling_1d_kernel(int8_t);
            break;
        }
        case TYPE_UINT8: {
            avg_pooling_1d_kernel(uint8_t);
            break;
        }
        case TYPE_INT16: {
            avg_pooling_1d_kernel(int16_t);
            break;
        }
        case TYPE_UINT16: {
            avg_pooling_1d_kernel(uint16_t);
            break;
        }
        case TYPE_INT32: {
            avg_pooling_1d_kernel(int32_t);
            break;
        }
        case TYPE_UINT32: {
            avg_pooling_1d_kernel(uint8_t);
            break;
        }
        case TYPE_INT64: {
            avg_pooling_1d_kernel(int64_t);
            break;
        }
        case TYPE_UINT64: {
            avg_pooling_1d_kernel(uint64_t);
            break;
        }
        case TYPE_FLOAT: {
            avg_pooling_1d_kernel(float);
            break;
        }
        case TYPE_DOUBLE: {
            avg_pooling_1d_kernel(double);
            break;
        }
        default:
            return STATUS_NOT_SUPPORTED;
    }

    return STATUS_SUCCESS;
}

static Status pooling_1d(const Tensor input, const char* mode,
                         const int *ksize, const int *stride,
                         const int *padding, Tensor *output){
    Status status;
    int64_t *in_dims = aitisa_tensor_dims(input);
    // calculate the dimensions of output
    int64_t out_dims[3];
    out_dims[0] = in_dims[0];
    out_dims[1] = in_dims[1];
    out_dims[2] = 1 + (in_dims[2] + padding[0] - ksize[0]) / stride[0];
    // create output
    CHECK_STATUS(aitisa_create(aitisa_tensor_data_type(input),
                               aitisa_tensor_device(input),
                               aitisa_tensor_layout_type(input),
                               out_dims, 3, output));

    if(!strcmp(mode, "max")){
        status = max_pooling_1d(input, ksize, stride, padding, output);
    }else if(!strcmp(mode, "avg")){
        status = avg_pooling_1d(input, ksize, stride, padding, output);
    }else{
        status = STATUS_NOT_SUPPORTED;
    }

    return status;
}



Status aitisa_pooling(const Tensor input, const char *mode,
                      const int *ksize, const int *stride,
                      const int *padding, Tensor *output){
    Status status;
    int64_t in_ndim = aitisa_tensor_ndim(input);
    switch(in_ndim){
        case 3:{
        // 1d-pooling
            status = pooling_1d(input, mode, ksize, stride, padding, output);
            break;
        }
        case 4:{
        // 2d-pooling

            break;
        }
        case 5:{
        // 3d-pooling

            break;
        }
        default:{
            status = STATUS_NOT_SUPPORTED;
        }
    }
    return status;
}


