#include "src/nn/pooling.h"
#include <math.h>
#include "src/basic/index_utils.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"

#define max_pooling_1d_kernel(typename)                                    \
  typename *turned_in_data = (typename *)in_data;                          \
  typename *turned_out_data = (typename *)out_data;                        \
  for (int64_t channel = 0; channel < nchannels_within_batch; channel++) { \
    int64_t out_offset = channel * out_feature_size;                       \
    int64_t in_offset = channel * in_feature_size;                         \
    int index_ktail =                                                      \
        ksize[0] + (dilation[0] - 1) * (ksize[0] - 1) - 1 - left_padding;  \
    for (int64_t out_feature = 0; out_feature < out_feature_size;          \
         out_feature++) {                                                  \
      typename max;                                                        \
      if (index_ktail < in_feature_size) {                                 \
        max = turned_in_data[in_offset + index_ktail];                     \
      } else {                                                             \
        max = 0;                                                           \
      }                                                                    \
      int index_khead =                                                    \
          index_ktail - (ksize[0] + (dilation[0] - 1) * (ksize[0] - 1));   \
      int dilation_flag = dilation[0];                                     \
      for (int i = index_ktail - 1; i > index_khead; i--) {                \
        if (dilation[0] > 1) {                                             \
          if (dilation_flag < dilation[0]) {                               \
            dilation_flag--;                                               \
            if (dilation_flag <= 0) {                                      \
              dilation_flag = dilation[0];                                 \
            }                                                              \
            continue;                                                      \
          }                                                                \
          dilation_flag--;                                                 \
        }                                                                  \
        /* search for the max value within a window */                     \
        if (i < 0) {                                                       \
          /* deal with left padding */                                     \
          if (max < 0) {                                                   \
            max = 0;                                                       \
          }                                                                \
          break;                                                           \
        }                                                                  \
        if (i >= in_feature_size) {                                        \
          /* deal with right padding */                                    \
          continue;                                                        \
        }                                                                  \
        if (turned_in_data[in_offset + i] > max) {                         \
          max = turned_in_data[in_offset + i];                             \
        }                                                                  \
      }                                                                    \
      turned_out_data[out_offset + out_feature] = max;                     \
      index_ktail += stride[0];                                            \
    }                                                                      \
  }

static Status max_pooling_1d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             const int *dilation, Tensor *output) {
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t *out_dims = aitisa_tensor_dims(*output);
  void *in_data = aitisa_tensor_data(input);
  int64_t in_feature_size = in_dims[2];
  void *out_data = aitisa_tensor_data(*output);
  int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
  int64_t out_feature_size = out_dims[2];
  int left_padding = padding[0] / 2;
  // printf("left padding is %d",left_padding);
  // int right_padding = padding[0] / 2 + (padding[0] % 2);

  switch (aitisa_tensor_data_type(input).code) {
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

#define avg_pooling_1d_kernel(typename)                                    \
  typename *turned_in_data = (typename *)in_data;                          \
  typename *turned_out_data = (typename *)out_data;                        \
  for (int64_t channel = 0; channel < nchannels_within_batch; channel++) { \
    int64_t out_offset = channel * out_feature_size;                       \
    int64_t in_offset = channel * in_feature_size;                         \
    int index_ktail =                                                      \
        ksize[0] + (dilation[0] - 1) * (ksize[0] - 1) - 1 - left_padding;  \
    for (int64_t out_feature = 0; out_feature < out_feature_size;          \
         out_feature++) {                                                  \
      typename total = 0;                                                  \
      int index_khead =                                                    \
          index_ktail - (ksize[0] + (dilation[0] - 1) * (ksize[0] - 1));   \
      int dilation_flag = dilation[0];                                     \
      for (int i = index_ktail; i > index_khead; i--) {                    \
        if (dilation[0] > 1) {                                             \
          if (dilation_flag < dilation[0]) {                               \
            dilation_flag--;                                               \
            if (dilation_flag <= 0) {                                      \
              dilation_flag = dilation[0];                                 \
            }                                                              \
            continue;                                                      \
          }                                                                \
          dilation_flag--;                                                 \
        }                                                                  \
        /* search for the max value within a window */                     \
        if (i < 0) {                                                       \
          /* deal with left padding */                                     \
          break;                                                           \
        }                                                                  \
        if (i >= in_feature_size) {                                        \
          /* deal with right padding */                                    \
          continue;                                                        \
        }                                                                  \
        total += turned_in_data[in_offset + i];                            \
      }                                                                    \
      turned_out_data[out_offset + out_feature] =                          \
          total / (typename)(ksize[0]);                                    \
      index_ktail += stride[0];                                            \
    }                                                                      \
  }

static Status avg_pooling_1d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             const int *dilation, Tensor *output) {
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t *out_dims = aitisa_tensor_dims(*output);
  void *in_data = aitisa_tensor_data(input);
  int64_t in_feature_size = in_dims[2];
  void *out_data = aitisa_tensor_data(*output);
  int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
  int64_t out_feature_size = out_dims[2];
  int left_padding = padding[0] / 2;
  // int right_padding = padding[0] / 2 + (padding[0] % 2);

  switch (aitisa_tensor_data_type(input).code) {
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

static Status pooling_1d(const Tensor input, const char *mode, const int *ksize,
                         const int *stride, const int *padding,
                         const int *dilation, Tensor *output) {
  Status status;
  int64_t *in_dims = aitisa_tensor_dims(input);
  // calculate the dimensions of output
  int64_t out_dims[3];
  out_dims[0] = in_dims[0];
  out_dims[1] = in_dims[1];
  out_dims[2] = 1 + (in_dims[2] + padding[0] - ksize[0] -
                     (dilation[0] - 1) * (ksize[0] - 1)) /
                        stride[0];
  // create output
  CHECK_STATUS(
      aitisa_create(aitisa_tensor_data_type(input), aitisa_tensor_device(input),
                    aitisa_tensor_layout_type(input), out_dims, 3, output));

  if (!strcmp(mode, "max")) {
    status = max_pooling_1d(input, ksize, stride, padding, dilation, output);
  } else if (!strcmp(mode, "avg")) {
    status = avg_pooling_1d(input, ksize, stride, padding, dilation, output);
  } else {
    status = STATUS_NOT_SUPPORTED;
  }

  return status;
}

#define max_pooling_2d_kernel(typename)                                        \
  typename *turned_in_data = (typename *)in_data;                              \
  typename *turned_out_data = (typename *)out_data;                            \
  for (int64_t channel = 0; channel < nchannels_within_batch; channel++) {     \
    int64_t out_offset = channel * out_feature_height * out_feature_width;     \
    int64_t in_offset = channel * in_feature_height * in_feature_width;        \
    int index_htail =                                                          \
        ksize[0] + (dilation[0] - 1) * (ksize[0] - 1) - 1 - up_padding;        \
    for (int64_t out_feature_h = 0; out_feature_h < out_feature_height;        \
         out_feature_h++) {                                                    \
      int index_wtail =                                                        \
          ksize[1] + (dilation[1] - 1) * (ksize[1] - 1) - 1 - left_padding;    \
      int out_h_offset = out_offset + out_feature_h * out_feature_width;       \
      int index_hhead =                                                        \
          index_htail - (ksize[0] + (dilation[0] - 1) * (ksize[0] - 1)) + 1;   \
      for (int64_t out_feature_w = 0; out_feature_w < out_feature_width;       \
           out_feature_w++) {                                                  \
        typename max;                                                          \
        int index_whead =                                                      \
            index_wtail - (ksize[1] + (dilation[1] - 1) * (ksize[1] - 1)) + 1; \
        if (index_whead < 0 || index_hhead < 0) {                              \
          max = 0;                                                             \
        } else if (index_whead >= in_feature_width ||                          \
                   index_hhead >= in_feature_height) {                         \
          max = 0;                                                             \
        } else {                                                               \
          max = turned_in_data[in_offset + index_hhead * in_feature_width +    \
                               index_whead];                                   \
        }                                                                      \
        int dilation_hflag = dilation[0];                                      \
        for (int in_feature_h = index_hhead; in_feature_h <= index_htail;      \
             in_feature_h++) {                                                 \
          if (dilation[0] > 1) {                                               \
            if (dilation_hflag < dilation[0]) {                                \
              dilation_hflag--;                                                \
              if (dilation_hflag <= 0) {                                       \
                dilation_hflag = dilation[0];                                  \
              }                                                                \
              continue;                                                        \
            }                                                                  \
            dilation_hflag--;                                                  \
          }                                                                    \
          /* search for the max value within a window */                       \
          if (in_feature_h < 0) continue;                                      \
          if (in_feature_h >= in_feature_height) break;                        \
          int in_h_offset = in_offset + in_feature_h * in_feature_width;       \
          int dilation_wflag = dilation[1];                                    \
          for (int in_feature_w = index_whead; in_feature_w <= index_wtail;    \
               in_feature_w++) {                                               \
            if (dilation[1] > 1) {                                             \
              if (dilation_wflag < dilation[1]) {                              \
                dilation_wflag--;                                              \
                if (dilation_wflag <= 0) {                                     \
                  dilation_wflag = dilation[1];                                \
                }                                                              \
                continue;                                                      \
              }                                                                \
              dilation_wflag--;                                                \
            }                                                                  \
            if (in_feature_w < 0) continue;                                    \
            if (in_feature_w >= in_feature_width) break;                       \
            if (max < turned_in_data[in_h_offset + in_feature_w]) {            \
              max = turned_in_data[in_h_offset + in_feature_w];                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        turned_out_data[out_h_offset + out_feature_w] = max;                   \
        index_wtail += stride[1];                                              \
      }                                                                        \
      index_htail += stride[0];                                                \
    }                                                                          \
  }

static Status max_pooling_2d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             const int *dilation, Tensor *output) {
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t *out_dims = aitisa_tensor_dims(*output);
  void *in_data = aitisa_tensor_data(input);
  int64_t in_feature_height = in_dims[2];
  int64_t in_feature_width = in_dims[3];
  void *out_data = aitisa_tensor_data(*output);
  int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
  int64_t out_feature_height = out_dims[2];
  int64_t out_feature_width = out_dims[3];
  int left_padding = padding[1] / 2;
  int up_padding = padding[0] / 2;

  switch (aitisa_tensor_data_type(input).code) {
    case TYPE_INT8: {
      max_pooling_2d_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      max_pooling_2d_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      max_pooling_2d_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      max_pooling_2d_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      max_pooling_2d_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      max_pooling_2d_kernel(uint8_t);
      break;
    }
    case TYPE_INT64: {
      max_pooling_2d_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      max_pooling_2d_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      max_pooling_2d_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      max_pooling_2d_kernel(double);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }

  return STATUS_SUCCESS;
}

#define avg_pooling_2d_kernel(typename)                                        \
  typename *turned_in_data = (typename *)in_data;                              \
  typename *turned_out_data = (typename *)out_data;                            \
  for (int64_t channel = 0; channel < nchannels_within_batch; channel++) {     \
    int64_t out_offset = channel * out_feature_height * out_feature_width;     \
    int64_t in_offset = channel * in_feature_height * in_feature_width;        \
    int index_htail =                                                          \
        ksize[0] + (dilation[0] - 1) * (ksize[0] - 1) - 1 - up_padding;        \
    for (int64_t out_feature_h = 0; out_feature_h < out_feature_height;        \
         out_feature_h++) {                                                    \
      int index_wtail =                                                        \
          ksize[1] + (dilation[1] - 1) * (ksize[1] - 1) - 1 - left_padding;    \
      int out_h_offset = out_offset + out_feature_h * out_feature_width;       \
      int index_hhead =                                                        \
          index_htail - (ksize[0] + (dilation[0] - 1) * (ksize[0] - 1)) + 1;   \
      for (int64_t out_feature_w = 0; out_feature_w < out_feature_width;       \
           out_feature_w++) {                                                  \
        typename total = 0;                                                    \
        int index_whead =                                                      \
            index_wtail - (ksize[1] + (dilation[1] - 1) * (ksize[1] - 1)) + 1; \
        int dilation_hflag = dilation[0];                                      \
        for (int in_feature_h = index_hhead; in_feature_h <= index_htail;      \
             in_feature_h++) {                                                 \
          if (dilation[0] > 1) {                                               \
            if (dilation_hflag < dilation[0]) {                                \
              dilation_hflag--;                                                \
              if (dilation_hflag <= 0) {                                       \
                dilation_hflag = dilation[0];                                  \
              }                                                                \
              continue;                                                        \
            }                                                                  \
            dilation_hflag--;                                                  \
          }                                                                    \
          /* calculate the average value within a window */                    \
          if (in_feature_h < 0) continue;                                      \
          if (in_feature_h >= in_feature_height) break;                        \
          int in_h_offset = in_offset + in_feature_h * in_feature_width;       \
          int dilation_wflag = dilation[1];                                    \
          for (int in_feature_w = index_whead; in_feature_w <= index_wtail;    \
               in_feature_w++) {                                               \
            if (dilation[1] > 1) {                                             \
              if (dilation_wflag < dilation[1]) {                              \
                dilation_wflag--;                                              \
                if (dilation_wflag <= 0) {                                     \
                  dilation_wflag = dilation[1];                                \
                }                                                              \
                continue;                                                      \
              }                                                                \
              dilation_wflag--;                                                \
            }                                                                  \
            if (in_feature_w < 0) continue;                                    \
            if (in_feature_w >= in_feature_width) break;                       \
            total += turned_in_data[in_h_offset + in_feature_w];               \
          }                                                                    \
        }                                                                      \
        turned_out_data[out_h_offset + out_feature_w] =                        \
            total / (typename)(ksize[0] * ksize[1]);                           \
        index_wtail += stride[1];                                              \
      }                                                                        \
      index_htail += stride[0];                                                \
    }                                                                          \
  }

static Status avg_pooling_2d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             const int *dilation, Tensor *output) {
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t *out_dims = aitisa_tensor_dims(*output);
  void *in_data = aitisa_tensor_data(input);
  int64_t in_feature_height = in_dims[2];
  int64_t in_feature_width = in_dims[3];
  void *out_data = aitisa_tensor_data(*output);
  int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
  int64_t out_feature_height = out_dims[2];
  int64_t out_feature_width = out_dims[3];
  int left_padding = padding[1] / 2;
  int up_padding = padding[0] / 2;

  switch (aitisa_tensor_data_type(input).code) {
    case TYPE_INT8: {
      avg_pooling_2d_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      avg_pooling_2d_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      avg_pooling_2d_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      avg_pooling_2d_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      avg_pooling_2d_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      avg_pooling_2d_kernel(uint8_t);
      break;
    }
    case TYPE_INT64: {
      avg_pooling_2d_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      avg_pooling_2d_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      avg_pooling_2d_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      avg_pooling_2d_kernel(double);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }

  return STATUS_SUCCESS;
}

static Status pooling_2d(const Tensor input, const char *mode, const int *ksize,
                         const int *stride, const int *padding,
                         const int *dilation, Tensor *output) {
  Status status;
  int64_t *in_dims = aitisa_tensor_dims(input);
  // calculate the dimensions of output
  int64_t out_dims[4];
  out_dims[0] = in_dims[0];
  out_dims[1] = in_dims[1];
  out_dims[2] = 1 + (in_dims[2] + padding[0] - ksize[0] -
                     (dilation[0] - 1) * (ksize[0] - 1)) /
                        stride[0];
  out_dims[3] = 1 + (in_dims[3] + padding[1] - ksize[1] -
                     (dilation[1] - 1) * (ksize[1] - 1)) /
                        stride[1];
  // create output
  CHECK_STATUS(
      aitisa_create(aitisa_tensor_data_type(input), aitisa_tensor_device(input),
                    aitisa_tensor_layout_type(input), out_dims, 4, output));

  if (!strcmp(mode, "max")) {
    status = max_pooling_2d(input, ksize, stride, padding, dilation, output);
  } else if (!strcmp(mode, "avg")) {
    status = avg_pooling_2d(input, ksize, stride, padding, dilation, output);
  } else {
    status = STATUS_NOT_SUPPORTED;
  }
  return status;
}

#define max_pooling_3d_kernel(typename)                                        \
  typename *turned_in_data = (typename *)in_data;                              \
  typename *turned_out_data = (typename *)out_data;                            \
  for (int64_t channel = 0; channel < nchannels_within_batch; channel++) {     \
    int64_t out_offset =                                                       \
        channel * out_feature_depth * out_feature_height * out_feature_width;  \
    int64_t in_offset =                                                        \
        channel * in_feature_depth * in_feature_height * in_feature_width;     \
    int index_dtail =                                                          \
        ksize[0] + (dilation[0] - 1) * (ksize[0] - 1) - 1 - front_padding;     \
    for (int64_t out_feature_d = 0; out_feature_d < out_feature_depth;         \
         out_feature_d++) {                                                    \
      int index_htail =                                                        \
          ksize[1] + (dilation[1] - 1) * (ksize[1] - 1) - 1 - up_padding;      \
      int index_dhead =                                                        \
          index_dtail - (ksize[0] + (dilation[0] - 1) * (ksize[0] - 1)) + 1;   \
      int out_d_offset =                                                       \
          out_offset + out_feature_d * out_feature_width * out_feature_height; \
      for (int64_t out_feature_h = 0; out_feature_h < out_feature_height;      \
           out_feature_h++) {                                                  \
        int index_wtail =                                                      \
            ksize[2] + (dilation[2] - 1) * (ksize[2] - 1) - 1 - left_padding;  \
        int out_h_offset = out_d_offset + out_feature_h * out_feature_width;   \
        int index_hhead =                                                      \
            index_htail - (ksize[1] + (dilation[1] - 1) * (ksize[1] - 1)) + 1; \
        for (int64_t out_feature_w = 0; out_feature_w < out_feature_width;     \
             out_feature_w++) {                                                \
          typename max;                                                        \
          int index_whead = index_wtail -                                      \
                            (ksize[2] + (dilation[2] - 1) * (ksize[2] - 1)) +  \
                            1;                                                 \
          if (index_whead < 0 || index_hhead < 0 || index_dhead < 0) {         \
            max = 0;                                                           \
          } else if (index_whead >= in_feature_width ||                        \
                     index_hhead >= in_feature_height ||                       \
                     index_dhead >= in_feature_depth) {                        \
            max = 0;                                                           \
          } else {                                                             \
            max =                                                              \
                turned_in_data[in_offset +                                     \
                               index_dhead * in_feature_height *               \
                                   in_feature_width +                          \
                               index_hhead * in_feature_width + index_whead];  \
          }                                                                    \
          int dilation_dflag = dilation[0];                                    \
          for (int in_feature_d = index_dhead; in_feature_d <= index_dtail;    \
               in_feature_d++) {                                               \
            if (dilation[0] > 1) {                                             \
              if (dilation_dflag < dilation[0]) {                              \
                dilation_dflag--;                                              \
                if (dilation_dflag <= 0) {                                     \
                  dilation_dflag = dilation[0];                                \
                }                                                              \
                continue;                                                      \
              }                                                                \
              dilation_dflag--;                                                \
            }                                                                  \
            /* search for the max value within a window */                     \
            if (in_feature_d < 0) continue;                                    \
            if (in_feature_d >= in_feature_depth) break;                       \
            int in_d_offset = in_offset + in_feature_d * in_feature_height *   \
                                              in_feature_width;                \
            int dilation_hflag = dilation[1];                                  \
            for (int in_feature_h = index_hhead; in_feature_h <= index_htail;  \
                 in_feature_h++) {                                             \
              if (dilation[1] > 1) {                                           \
                if (dilation_hflag < dilation[1]) {                            \
                  dilation_hflag--;                                            \
                  if (dilation_hflag <= 0) {                                   \
                    dilation_hflag = dilation[1];                              \
                  }                                                            \
                  continue;                                                    \
                }                                                              \
                dilation_hflag--;                                              \
              }                                                                \
              if (in_feature_h < 0) continue;                                  \
              if (in_feature_h >= in_feature_height) break;                    \
              int in_h_offset = in_d_offset + in_feature_h * in_feature_width; \
              int dilation_wflag = dilation[2];                                \
              for (int in_feature_w = index_whead;                             \
                   in_feature_w <= index_wtail; in_feature_w++) {              \
                if (dilation[2] > 1) {                                         \
                  if (dilation_wflag < dilation[2]) {                          \
                    dilation_wflag--;                                          \
                    if (dilation_wflag <= 0) {                                 \
                      dilation_wflag = dilation[2];                            \
                    }                                                          \
                    continue;                                                  \
                  }                                                            \
                  dilation_wflag--;                                            \
                }                                                              \
                if (in_feature_w < 0) continue;                                \
                if (in_feature_w >= in_feature_width) break;                   \
                if (max < turned_in_data[in_h_offset + in_feature_w]) {        \
                  max = turned_in_data[in_h_offset + in_feature_w];            \
                }                                                              \
              }                                                                \
            }                                                                  \
          }                                                                    \
                                                                               \
          turned_out_data[out_h_offset + out_feature_w] = max;                 \
          index_wtail += stride[2];                                            \
        }                                                                      \
        index_htail += stride[1];                                              \
      }                                                                        \
      index_dtail += stride[0];                                                \
    }                                                                          \
  }

static Status max_pooling_3d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             const int *dilation, Tensor *output) {
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t *out_dims = aitisa_tensor_dims(*output);
  void *in_data = aitisa_tensor_data(input);
  int64_t in_feature_depth = in_dims[2];
  int64_t in_feature_height = in_dims[3];
  int64_t in_feature_width = in_dims[4];
  void *out_data = aitisa_tensor_data(*output);
  int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
  int64_t out_feature_depth = out_dims[2];
  int64_t out_feature_height = out_dims[3];
  int64_t out_feature_width = out_dims[4];
  int left_padding = padding[2] / 2;
  int up_padding = padding[1] / 2;
  int front_padding = padding[0] / 2;

  switch (aitisa_tensor_data_type(input).code) {
    case TYPE_INT8: {
      max_pooling_3d_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      max_pooling_3d_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      max_pooling_3d_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      max_pooling_3d_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      max_pooling_3d_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      max_pooling_3d_kernel(uint8_t);
      break;
    }
    case TYPE_INT64: {
      max_pooling_3d_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      max_pooling_3d_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      max_pooling_3d_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      max_pooling_3d_kernel(double);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }
  return STATUS_SUCCESS;
}

#define avg_pooling_3d_kernel(typename)                                        \
  typename *turned_in_data = (typename *)in_data;                              \
  typename *turned_out_data = (typename *)out_data;                            \
  for (int64_t channel = 0; channel < nchannels_within_batch; channel++) {     \
    int64_t out_offset =                                                       \
        channel * out_feature_depth * out_feature_height * out_feature_width;  \
    int64_t in_offset =                                                        \
        channel * in_feature_depth * in_feature_height * in_feature_width;     \
    int index_dtail =                                                          \
        ksize[0] + (dilation[0] - 1) * (ksize[0] - 1) - 1 - front_padding;     \
    for (int64_t out_feature_d = 0; out_feature_d < out_feature_depth;         \
         out_feature_d++) {                                                    \
      int index_htail =                                                        \
          ksize[1] + (dilation[1] - 1) * (ksize[1] - 1) - 1 - up_padding;      \
      int index_dhead =                                                        \
          index_dtail - (ksize[0] + (dilation[0] - 1) * (ksize[0] - 1)) + 1;   \
      int out_d_offset =                                                       \
          out_offset + out_feature_d * out_feature_width * out_feature_height; \
      for (int64_t out_feature_h = 0; out_feature_h < out_feature_height;      \
           out_feature_h++) {                                                  \
        int index_wtail =                                                      \
            ksize[2] + (dilation[2] - 1) * (ksize[2] - 1) - 1 - left_padding;  \
        int out_h_offset = out_d_offset + out_feature_h * out_feature_width;   \
        int index_hhead =                                                      \
            index_htail - (ksize[1] + (dilation[1] - 1) * (ksize[1] - 1)) + 1; \
        for (int64_t out_feature_w = 0; out_feature_w < out_feature_width;     \
             out_feature_w++) {                                                \
          typename total = 0;                                                  \
          int index_whead = index_wtail -                                      \
                            (ksize[2] + (dilation[2] - 1) * (ksize[2] - 1)) +  \
                            1;                                                 \
          int dilation_dflag = dilation[0];                                    \
          for (int in_feature_d = index_dhead; in_feature_d <= index_dtail;    \
               in_feature_d++) {                                               \
            if (dilation[0] > 1) {                                             \
              if (dilation_dflag < dilation[0]) {                              \
                dilation_dflag--;                                              \
                if (dilation_dflag <= 0) {                                     \
                  dilation_dflag = dilation[0];                                \
                }                                                              \
                continue;                                                      \
              }                                                                \
              dilation_dflag--;                                                \
            }                                                                  \
            /* search for the avg value within a window */                     \
            if (in_feature_d < 0) continue;                                    \
            if (in_feature_d >= in_feature_depth) break;                       \
            int in_d_offset = in_offset + in_feature_d * in_feature_height *   \
                                              in_feature_width;                \
            int dilation_hflag = dilation[1];                                  \
            for (int in_feature_h = index_hhead; in_feature_h <= index_htail;  \
                 in_feature_h++) {                                             \
              if (dilation[1] > 1) {                                           \
                if (dilation_hflag < dilation[1]) {                            \
                  dilation_hflag--;                                            \
                  if (dilation_hflag <= 0) {                                   \
                    dilation_hflag = dilation[1];                              \
                  }                                                            \
                  continue;                                                    \
                }                                                              \
                dilation_hflag--;                                              \
              }                                                                \
              if (in_feature_h < 0) continue;                                  \
              if (in_feature_h >= in_feature_height) break;                    \
              int in_h_offset = in_d_offset + in_feature_h * in_feature_width; \
              int dilation_wflag = dilation[2];                                \
              for (int in_feature_w = index_whead;                             \
                   in_feature_w <= index_wtail; in_feature_w++) {              \
                if (dilation[2] > 1) {                                         \
                  if (dilation_wflag < dilation[2]) {                          \
                    dilation_wflag--;                                          \
                    if (dilation_wflag <= 0) {                                 \
                      dilation_wflag = dilation[2];                            \
                    }                                                          \
                    continue;                                                  \
                  }                                                            \
                  dilation_wflag--;                                            \
                }                                                              \
                if (in_feature_w < 0) continue;                                \
                if (in_feature_w >= in_feature_width) break;                   \
                total += turned_in_data[in_h_offset + in_feature_w];           \
              }                                                                \
            }                                                                  \
          }                                                                    \
          turned_out_data[out_h_offset + out_feature_w] =                      \
              total / (typename)(ksize[0] * ksize[1] * ksize[2]);              \
          index_wtail += stride[2];                                            \
        }                                                                      \
        index_htail += stride[1];                                              \
      }                                                                        \
      index_dtail += stride[0];                                                \
    }                                                                          \
  }

static Status avg_pooling_3d(const Tensor input, const int *ksize,
                             const int *stride, const int *padding,
                             const int *dilation, Tensor *output) {
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t *out_dims = aitisa_tensor_dims(*output);
  void *in_data = aitisa_tensor_data(input);
  int64_t in_feature_depth = in_dims[2];
  int64_t in_feature_height = in_dims[3];
  int64_t in_feature_width = in_dims[4];
  void *out_data = aitisa_tensor_data(*output);
  int64_t nchannels_within_batch = out_dims[0] * out_dims[1];
  int64_t out_feature_depth = out_dims[2];
  int64_t out_feature_height = out_dims[3];
  int64_t out_feature_width = out_dims[4];
  int left_padding = padding[2] / 2;
  int up_padding = padding[1] / 2;
  int front_padding = padding[0] / 2;

  switch (aitisa_tensor_data_type(input).code) {
    case TYPE_INT8: {
      avg_pooling_3d_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      avg_pooling_3d_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      avg_pooling_3d_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      avg_pooling_3d_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      avg_pooling_3d_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      avg_pooling_3d_kernel(uint8_t);
      break;
    }
    case TYPE_INT64: {
      avg_pooling_3d_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      avg_pooling_3d_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      avg_pooling_3d_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      avg_pooling_3d_kernel(double);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }
  return STATUS_SUCCESS;
}

static Status pooling_3d(const Tensor input, const char *mode, const int *ksize,
                         const int *stride, const int *padding,
                         const int *dilation, Tensor *output) {
  Status status;
  int64_t *in_dims = aitisa_tensor_dims(input);
  // calculate the dimensions of output
  int64_t out_dims[5];
  out_dims[0] = in_dims[0];
  out_dims[1] = in_dims[1];
  out_dims[2] = 1 + (in_dims[2] + padding[0] - ksize[0] -
                     (dilation[0] - 1) * (ksize[0] - 1)) /
                        stride[0];
  out_dims[3] = 1 + (in_dims[3] + padding[1] - ksize[1] -
                     (dilation[1] - 1) * (ksize[1] - 1)) /
                        stride[1];
  out_dims[4] = 1 + (in_dims[4] + padding[2] - ksize[2] -
                     (dilation[2] - 1) * (ksize[2] - 1)) /
                        stride[2];
  // create output
  CHECK_STATUS(
      aitisa_create(aitisa_tensor_data_type(input), aitisa_tensor_device(input),
                    aitisa_tensor_layout_type(input), out_dims, 5, output));

  if (!strcmp(mode, "max")) {
    status = max_pooling_3d(input, ksize, stride, padding, dilation, output);
  } else if (!strcmp(mode, "avg")) {
    status = avg_pooling_3d(input, ksize, stride, padding, dilation, output);
  } else {
    status = STATUS_NOT_SUPPORTED;
  }
  return status;
}

Status aitisa_pooling(const Tensor input, const char *mode, const int *ksize,
                      const int *stride, const int *padding,
                      const int *dilation, Tensor *output) {
  Status status;
  int64_t in_ndim = aitisa_tensor_ndim(input);
  switch (in_ndim) {
    case 3: {
      // 1d-pooling
      status =
          pooling_1d(input, mode, ksize, stride, padding, dilation, output);
      break;
    }
    case 4: {
      // 2d-pooling
      status =
          pooling_2d(input, mode, ksize, stride, padding, dilation, output);
      break;
    }
    case 5: {
      // 3d-pooling
      status =
          pooling_3d(input, mode, ksize, stride, padding, dilation, output);
      break;
    }
    default:
      status = STATUS_NOT_SUPPORTED;
  }
  return status;
}
