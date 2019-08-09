#include <stdio.h>
#include "../core/tensor.h"

void aitisa_add(const Tensor a, const Tensor b, Tensor *output){
    //check if Tensor a has the same shape as the shape of Tensor b
    if(a.shape->ndim != b.shape->ndim){
        printf("ndims are not equal!\n");
        exit(0);
    }
    for(int i=0; i<a.shape->ndim; i++){
        if(a.shape->dims[i] != b.shape->dims[i]){
            printf("dims[%d] are not equal!\n", i);
            exit(0);
        }
    }

    //check if a and b have the same data type
    if(a.storage->dtype->code != b.storage->dtype->code){
        printf("a and b do not have the same ScalarType!\n");
        exit(0);
    }

    // create Tensor output
    aitisa_create(*(a.storage->dtype), *(a.shape->layout),
                  *(a.storage->device), a.shape->dims,
                  a.shape->ndim, output);

    //kernel
    switch(a.storage->dtype->code){
        case 0:{
            int8_t *a_data = (int8_t*)a.storage->data;
            int8_t *b_data = (int8_t*)b.storage->data;
            int8_t *output_data = (int8_t*)malloc(a.storage->size * sizeof(int8_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 1:{
            int16_t *a_data = (int16_t*)a.storage->data;
            int16_t *b_data = (int16_t*)b.storage->data;
            int16_t *output_data = (int16_t*)malloc(a.storage->size * sizeof(int16_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 2:{
            int32_t *a_data = (int32_t*)a.storage->data;
            int32_t *b_data = (int32_t*)b.storage->data;
            int32_t *output_data = (int32_t*)malloc(a.storage->size * sizeof(int32_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 3:{
            int64_t *a_data = (int64_t*)a.storage->data;
            int64_t *b_data = (int64_t*)b.storage->data;
            int64_t *output_data = (int64_t*)malloc(a.storage->size * sizeof(int64_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 4:{
            uint8_t *a_data = (uint8_t*)a.storage->data;
            uint8_t *b_data = (uint8_t*)b.storage->data;
            uint8_t *output_data = (uint8_t*)malloc(a.storage->size * sizeof(uint8_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 5:{
            uint16_t *a_data = (uint16_t*)a.storage->data;
            uint16_t *b_data = (uint16_t*)b.storage->data;
            uint16_t *output_data = (uint16_t*)malloc(a.storage->size * sizeof(uint16_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 6:{
            uint32_t *a_data = (uint32_t*)a.storage->data;
            uint32_t *b_data = (uint32_t*)b.storage->data;
            uint32_t *output_data = (uint32_t*)malloc(a.storage->size * sizeof(uint32_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 7:{
            uint64_t *a_data = (uint64_t*)a.storage->data;
            uint64_t *b_data = (uint64_t*)b.storage->data;
            uint64_t *output_data = (uint64_t*)malloc(a.storage->size * sizeof(uint64_t));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 8:{
            float *a_data = (float*)a.storage->data;
            float *b_data = (float*)b.storage->data;
            float *output_data = (float*)malloc(a.storage->size * sizeof(float));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        case 9:{
            double *a_data = (double*)a.storage->data;
            double *b_data = (double*)b.storage->data;
            double *output_data = (double*)malloc(a.storage->size * sizeof(double));
            for(int i=0; i<a.storage->size; i++){
                output_data[i] = a_data[i] + b_data[i];
            }
            output->storage->data = (void*)output_data;
            break;
        }
        default: break;
    }
}
