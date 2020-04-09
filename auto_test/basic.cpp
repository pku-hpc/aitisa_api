#include "auto_test/basic.h"
#include <random>

namespace aitisa_api {

const DataType aitisa_dtypes[10] = {kInt8,   kUint8, 
                                    kInt16,  kUint16, 
                                    kInt32,  kUint32, 
                                    kInt64,  kUint64, 
                                    kFloat,  kDouble};
const Device aitisa_devices[2] = { {DEVICE_CPU,  0}, 
                                   {DEVICE_CUDA, 0} };

template <typename DATATYPE>
inline void natural_assign_int(DATATYPE *data, unsigned int nelem){
  for(unsigned int i=0; i<nelem; i++){
    data[i] = (DATATYPE)i + 1;
  }
}
template <typename DATATYPE>
inline void natural_assign_float(DATATYPE *data, unsigned int nelem){
  for(unsigned int i=0; i<nelem; i++){
    data[i] = (DATATYPE)i * 0.1 + 0.1;
  }
}
void natural_assign(void *data, unsigned int len, int dtype){
  switch (dtype) {
    case 0: natural_assign_int((int8_t*)data, len/sizeof(int8_t)); break;
    case 1: natural_assign_int((uint8_t*)data, len/sizeof(uint8_t)); break;
    case 2: natural_assign_int((int16_t*)data, len/sizeof(int16_t)); break;
    case 3: natural_assign_int((uint16_t*)data, len/sizeof(uint16_t)); break;
    case 4: natural_assign_int((int32_t*)data, len/sizeof(int32_t)); break;
    case 5: natural_assign_int((uint32_t*)data, len/sizeof(uint32_t)); break;
    case 6: natural_assign_int((int64_t*)data, len/sizeof(int64_t)); break;
    case 7: natural_assign_int((uint64_t*)data, len/sizeof(uint64_t)); break;
    case 8: natural_assign_float((float*)data, len/sizeof(float)); break;
    case 9: natural_assign_float((double*)data, len/sizeof(double)); break;
    default: break;
  }
}

template <typename DATATYPE>
inline void random_assign_int(DATATYPE *data, unsigned int nelem){
  std::default_random_engine gen(/*seed*/0);
  std::uniform_int_distribution<DATATYPE> dis(0, 10);
  for(unsigned int i=0; i<nelem; i++){
    data[i] = dis(gen);
  }
}
template <typename DATATYPE>
inline void random_assign_float(DATATYPE *data, unsigned int nelem){
  std::default_random_engine gen(/*seed*/0);
  std::normal_distribution<DATATYPE> dis(0,1);
  for(unsigned int i=0; i<nelem; i++){
    data[i] = dis(gen);
  }
}
void random_assign(void *data, unsigned int len, int dtype){
  switch (dtype) {
    case 0: random_assign_int((int8_t*)data, len/sizeof(int8_t)); break;
    case 1: random_assign_int((uint8_t*)data, len/sizeof(uint8_t)); break;
    case 2: random_assign_int((int16_t*)data, len/sizeof(int16_t)); break;
    case 3: random_assign_int((uint16_t*)data, len/sizeof(uint16_t)); break;
    case 4: random_assign_int((int32_t*)data, len/sizeof(int32_t)); break;
    case 5: random_assign_int((uint32_t*)data, len/sizeof(uint32_t)); break;
    case 6: random_assign_int((int64_t*)data, len/sizeof(int64_t)); break;
    case 7: random_assign_int((uint64_t*)data, len/sizeof(uint64_t)); break;
    case 8: random_assign_float((float*)data, len/sizeof(float)); break;
    case 9: random_assign_float((double*)data, len/sizeof(double)); break;
    default: break;
  }
}

} // namespace aitisa_api