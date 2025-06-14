//task 2
#include <cstdio>
#include <iostream>
#include <cstdint>
#include <climits>
#include <chrono>
#include <random>
#include <curand_kernel.h>
static const uint32_t prime32_1 = 2654435761U;
static const uint32_t prime32_2 = 2246822519U;
static const uint32_t prime32_3 = 3266489917U;
static const uint32_t prime32_4 = 668265263U;
__managed__ bool legal = true;
__host__ __device__ uint32_t Hash(uint32_t key, uint32_t seed,uint32_t size) {
    uint32_t hash = seed + prime32_4;
    hash += key * prime32_2;
    hash = (hash << 13) | (hash >> (32 - 13));
    hash *= prime32_1;
    hash = (hash ^ (hash >> 15)) * prime32_2;
    hash = (hash ^ (hash >> 13)) * prime32_3;
    hash = hash ^ (hash >> 16);
    return hash%size;
}
__global__ void insert_2_kernel (uint32_t *table, uint32_t table_size, uint32_t *keys, uint32_t key_size, int MAX_step, uint32_t seed1, uint32_t seed2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= key_size) return;
    uint32_t key = keys[idx];
    uint32_t h1 = Hash(key,seed1,table_size), h2 = Hash(key,seed2,table_size);
    if (table[h1] == key || table[h2] == key) return;
    key = atomicExch(&table[h1], key);
    uint32_t evict = h1;
    if (key == UINT_MAX) return;
    for(int i = 0; i < MAX_step; i++){
        h1 = Hash(key,seed1,table_size), h2 = Hash(key,seed2,table_size);
        if (table[h1] == key || table[h2] == key) return;
        if(evict == h1) evict = h2;
        else evict = h1;
        key = atomicExch(&table[evict], key);
        if (key == UINT_MAX) return;
    }
    legal = false;
}
__global__ void insert_3_kernel (uint32_t *table, uint32_t table_size, uint32_t *keys, uint32_t key_size, int MAX_step, uint32_t seed1, uint32_t seed2,uint32_t seed3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= key_size) return;
    uint32_t key = keys[idx];
    uint32_t h1 = Hash(key,seed1,table_size), h2 = Hash(key,seed2,table_size), h3 = Hash(key,seed3,table_size);
    if (table[h1] == key || table[h2] == key || table[h3] == key) return;
    uint32_t evict = h1;
    key = atomicExch(&table[h1], key);
    if (key == UINT_MAX) return;
    for(int i = 0; i < MAX_step; i++){
        h1 = Hash(key,seed1,table_size), h2 = Hash(key,seed2,table_size), h3 = Hash(key,seed3,table_size);
        if (table[h1] == key || table[h2] == key || table[h3] == key) return;
        if(evict == h1) evict = h2;
        else if(evict == h2) evict = h3;
        else evict = h1;
        key = atomicExch(&table[evict], key);
        if (key == UINT_MAX) return;
    } 
    legal = false;
}
__global__ void generate_key_kernel(uint32_t *keys, uint32_t key_size, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= key_size) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    keys[idx] = curand(&state);
}
__global__ void generate_lookup_key_kernel(uint32_t *keys, uint32_t key_size, uint32_t seed, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= key_size || idx % 10 < i) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    keys[idx] = curand(&state);
}

__global__ void find_2_kernel (uint32_t *table, uint32_t table_size, uint32_t *keys, uint32_t key_size, int MAX_step, uint32_t seed1, uint32_t seed2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= key_size) return;
    uint32_t key = keys[idx];
    uint32_t h1 = Hash(key,seed1,table_size), h2 = Hash(key,seed2,table_size);
    if (table[h1] == key || table[h2] == key) {
        if (table[h1] == key) keys[idx] = h1;
        if (table[h2] == key) keys[idx] = h2;
    }
    else keys[idx] = UINT_MAX;
    return;
}
__global__ void find_3_kernel (uint32_t *table, uint32_t table_size, uint32_t *keys, uint32_t key_size, int MAX_step, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= key_size) return;
    uint32_t key = keys[idx];
    uint32_t h1 = Hash(key,seed1,table_size), h2 = Hash(key,seed2,table_size), h3 = Hash(key,seed3,table_size);
    if (table[h1] == key || table[h2] == key || table[h3] == key) {
        if (table[h1] == key) keys[idx] = h1;
        if (table[h2] == key) keys[idx] = h2;
        if (table[h3] == key) keys[idx] = h3;
    }
    else keys[idx] = UINT_MAX;
    return;
}
//const uint32_t MAX_table_size = 1<<25;
//const uint32_t MAX_key_size = 1<<24;
int main() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    std::cout<<"CUDA Runtime Version: " << version / 1000 << "." << (version % 1000) / 10 << std::endl;
    uint32_t MAX_step = 96;
    // int *h_table = new int[MAX_table_size];
    // int *h_key = new int[MAX_key_size];
    uint32_t key_size = 1<<24;
    int table_size = 1<<25;
    printf("table_size = %d\n",table_size);
    int block_size = 512;
    int grid_size = (key_size + block_size - 1) / block_size;
    srand(time(NULL));
    uint32_t *table, *keys;
    printf("2 hash functions\n");
    for(int i=0;i<=10;i++){
        grid_size = (key_size + block_size - 1) / block_size;
        cudaError_t err;
        err=cudaMallocManaged((void**)&table,  sizeof(uint32_t)*table_size);
        // cudaMemcpy(table, h_table, sizeof(uint32_t)*table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA malloc failed: %s\n",cudaGetErrorString(err)) ;
            return -1;
        }   
        // cudaMemcpy(keys, h_key, sizeof(uint32_t)*key_size, cudaMemcpyHostToDevice);
        
        err=cudaMallocManaged((void**)&keys,  sizeof(uint32_t)*key_size);
        if (err != cudaSuccess) {
            printf("CUDA malloc failed for keys\n");
            return -1;
        }
        cudaMemset(table,0xff,table_size * sizeof(uint32_t));
        uint32_t seed = rand(), seed1 = rand(), seed2 = rand(), seed_lookup = rand();
        generate_key_kernel<<<grid_size, block_size>>>(keys, key_size, seed);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        legal = true;
        insert_2_kernel<<<grid_size, block_size>>>(table, table_size, keys, key_size, MAX_step, seed1, seed2);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        if(legal) printf("all keys are inserted ");
        else printf("some keys are not inserted ");
        generate_lookup_key_kernel<<<grid_size, block_size>>>(keys, key_size, seed_lookup, i);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        auto time1 = std::chrono::high_resolution_clock::now();
        find_2_kernel<<<grid_size, block_size>>>(table, table_size, keys, key_size, MAX_step, seed1, seed2);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        auto time2 = std::chrono::high_resolution_clock::now();
        long long time = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
        printf("s_%d time=%lld MOPS=%lf\n", i, time, 1.0 * key_size / time);
        cudaFree(table);
        cudaFree(keys);
    }
    printf("3 hash functions\n");
    for(int i=0;i<=10;i++){
        grid_size = (key_size + block_size - 1) / block_size;
        cudaError_t err;
        err=cudaMallocManaged((void**)&table,  sizeof(uint32_t)*table_size);
        // cudaMemcpy(table, h_table, sizeof(uint32_t)*table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA malloc failed: %s\n",cudaGetErrorString(err)) ;
            return -1;
        }   
        // cudaMemcpy(keys, h_key, sizeof(uint32_t)*key_size, cudaMemcpyHostToDevice);
        err=cudaMallocManaged((void**)&keys,  sizeof(uint32_t)*key_size);
        if (err != cudaSuccess) {
            printf("CUDA malloc failed for keys\n");
            return -1;
        }
        cudaMemset(table,0xff,table_size * sizeof(uint32_t));
        uint32_t seed = rand(), seed1 = rand(), seed2 = rand(), seed_lookup = rand(), seed3 = rand();
        generate_key_kernel<<<grid_size, block_size>>>(keys, key_size, seed);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        legal = true;
        insert_3_kernel<<<grid_size, block_size>>>(table, table_size, keys, key_size, MAX_step, seed1, seed2, seed3);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        if(legal) printf("all keys are inserted ");
        else printf("some keys are not inserted ");
        generate_lookup_key_kernel<<<grid_size, block_size>>>(keys, key_size, seed_lookup, i);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        auto time1 = std::chrono::high_resolution_clock::now();
        find_3_kernel<<<grid_size, block_size>>>(table, table_size, keys, key_size, MAX_step, seed1, seed2, seed3);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        auto time2 = std::chrono::high_resolution_clock::now();
        long long time = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
        printf("s_%d time=%lld MOPS=%lf\n", i, time, 1.0 * key_size / time);
        cudaFree(table);
        cudaFree(keys);
    }
    return 0;
}