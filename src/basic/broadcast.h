#ifndef BROADCAST_H
#define BROADCAST_H

#include "src/core/tensor.h"

/**
 * @brief Do broadcast and save result into dims_out.
 *
 * @details Compares every pair of elements of dims_in1 and dims_in2 in
 * 			reverse order.
 * 			Suppose that k and m are the i_th last elements of dims_in1 and
 * 			dims_in2. (1) k == m, then dim_out[i_th last] = k; (2) k == 1 or m ==1, then
 * 			dim_out[i_th last] = max(k, m); (3) k != m and k != 1 and m !=1, mismatched error.
 * 			For example: dims_in1 = {3, 2, 3, 1, 5}, dims_in2 = {1, 3, 4, 5}, then
 * 			dims_out = {3, 2, 3, 4, 5}
 *
 * @note This function does not allocated any memory inside, so the memory of
 * dims_out is assumed to be managed by user.
 *
 * @param dims_in1 The first dims.
 * @param ndim_in1 The size of dims_in1.
 * @param dims_in2 The second dims.
 * @param ndim_in2 The size of dims_in2.
 * @param dims_out The output dims.
 * @param ndim_out The size of dims_out.
 *
 * @return Status.
 * @retval STATUS_SUCCESS Success.
 * @retval STATUS_DIMENSIONS_MISMATCH The dims of input are mismatched and can not be broadcasted.
 */
AITISA_API_PUBLIC Status
aitisa_broadcast_array(int64_t* dims_in1, int64_t ndim_in1, int64_t* dims_in2,
                       int64_t ndim_in2, int64_t* dims_out, int64_t ndim_out);

#endif
