#include "src/core/allocator.h"
#include "src/nn/dropout.h"
#include "src/basic/duplicate.h"
#include<time.h>
#include<stdlib.h>

static Status dropout_rand_array(int64_t begin, int64_t end,
                                 int64_t amount, int64_t **recorder){
  *recorder =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(**recorder)*amount);
  if(!*recorder){
    return STATUS_ALLOC_FAILED;
  }
  srand(time(NULL));
  int64_t generated_amount = 0;
  double expected_rate = amount / (end - begin + 1);
  for(int64_t i=0; i<end-begin+1; i++){
    int64_t rand_num = (rand() % (end - begin + 1)) + 1;
    if(rand_num <= amount){
      generated_amount++;
      (*recorder)[generated_amount-1] = i + begin;
      if(generated_amount > amount) break;
    }
    double present_rate = (double)generated_amount / (double)(i + 1);
    if(expected_rate - present_rate > 0){
      for(int64_t j=i+1; j<end-begin+1; j++){
        generated_amount++;
        (*recorder)[generated_amount-1] = j + begin;
        if(expected_rate - present_rate <= 0){
          i = j;
          break;
        }
        if(generated_amount > amount) break;
      }
      if(generated_amount > amount) break;
    }
  }
  if(generated_amount < amount){
    int64_t gap = generated_amount;
    for(int64_t i=end-begin; i>=0; i--){
      // judge whether i+begin is already in recorder
      int flag = 0;
      for(int64_t j=gap-1; j>=0; j--){
        if((*recorder)[j] < i+begin) break;
        if((*recorder)[j] == i+begin){
          flag = 1;
          break;
        }
      }
      // if i+begin is not in recorder, then pull it into recorder
      if(flag == 0){
        generated_amount++;
        (*recorder)[generated_amount-1] = i + begin;
      }
      if(generated_amount >= amount) break;
    }
  }
  return STATUS_SUCCESS;
}

#define dropout_kernel(typename)                              \
  typename* data = (typename *)aitisa_tensor_data(*tensor);   \
  for(int64_t i=0; i<amount; i++){                            \
    int64_t idx = recorder[i];                                \
    data[idx] = 0;                                            \
  }

static Status dropout_template(Tensor *tensor, const double rate){
  int64_t size = aitisa_tensor_size(*tensor);
  int64_t amount = (int64_t)(size * rate);
  // generate an unrepeated integer array with length of size
  int64_t *recorder;
  CHECK_STATUS(
    dropout_rand_array(0, size-1, amount, &recorder));
  // implement dropout kernel
  DataType dtype = aitisa_tensor_data_type(*tensor);
  Status status = STATUS_SUCCESS;
  switch(dtype.code){
    case TYPE_INT8: {
      dropout_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      dropout_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      dropout_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      dropout_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      dropout_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      dropout_kernel(uint32_t);
      break;
    }
    case TYPE_INT64: {
      dropout_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      dropout_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      dropout_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      dropout_kernel(double);
      break;
    }
    default:
      status = STATUS_NOT_SUPPORTED;
  }
  aitisa_default_cpu_allocator()->raw_dealloc(recorder);
  return status;
}

Status aitisa_dropout(const Tensor input, const double rate,
                      Tensor *output){
  // check if rate satisfy 0 <= rate <= 1
  if(rate<0 || rate>1){
    return STATUS_INVALID_ARGUMENT;
  }
  // copy input
  Tensor new_tensor;
  CHECK_STATUS(
    aitisa_duplicate(input, &new_tensor));
  // implement dropout
  CHECK_STATUS(
    dropout_template(&new_tensor, rate));
  *output = new_tensor;
  return STATUS_SUCCESS;
}
