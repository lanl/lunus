
#define CUB_STDERR                                // ADDED -Pierre   line 2

//#include <stdio.h>                                // ADDED -Pierre   line 6
//#include <algorithm>                              // ADDED -Pierre   line 7

#include <cub/util_allocator.cuh>                 // ADDED -Pierre   line 9
#include <cub/device/device_radix_sort.cuh>       // ADDED -Pierre   line 10

#include <cub/device/device_segmented_radix_sort.cuh>     // SEGMENTED ADDED  -Pierre

//#include "test/test_util.h"                       // ADDED -Pierre   line 13

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


////////////////////////////////////////////////
// CUB related functions BEGIN
////////////////////////////////////////////////
/**
 * Simple key pairing for floating point types.  Distinguishes
 * between positive and negative zero.
 */
/*
struct Pair
{
    float   key;

    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;

        if (key > b.key)
            return false;

        // Return true if key is negative zero and b.key is positive zero
        unsigned int key_bits   = *reinterpret_cast<unsigned*>(const_cast<float*>(&key));
        unsigned int b_key_bits = *reinterpret_cast<unsigned*>(const_cast<float*>(&b.key));
        unsigned int HIGH_BIT   = 1u << 31;

        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
    }
};

*/
/**
 * Initialize key sorting problem.
 */
/*
void Initialize(
    float           *h_keys,
    float           *h_reference_keys,
    int             num_items)
{
    Pair *h_pairs = new Pair[num_items];

    for (int i = 0; i < num_items; ++i)
    {
        RandomBits(h_keys[i]);
        h_pairs[i].key    = h_keys[i];
    }

    if (g_verbose)
    {

       printf("Input keys:\n");
       DisplayResults(h_keys, num_items);
       printf("\n\n");

    }

    std::stable_sort(h_pairs, h_pairs + num_items);

    for (int i = 0; i < num_items; ++i)
    {
        h_reference_keys[i]     = h_pairs[i].key;
    }

    delete[] h_pairs;
}
*/

extern "C" void quickSortListCUB(size_t arr[], size_t stack[], size_t num_arrays, size_t array_size)
{
  size_t i;

 // int num_items = array_size-1;     // CUB <--- quickSortIterative
    // SEGMENTED SNIPPET from : https://nvlabs.github.io/cub/structcub_1_1_device_segmented_radix_sort.html#a175a8f431517e609e3d683076de86402
  int num_segments =(int)num_arrays;                                         // CUB segmented Radix Sort
  int   *h_offsets             = new int[num_segments+1];
  int   *d_offsets;
  size_t *d_keys_in;
  size_t *d_keys_out;
//  size_t *h_keys_out;
  
  // Allocate device arrays
//  DoubleBuffer<unsigned long> d_keys;
//  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(unsigned long) * array_size));
//  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(unsigned long) * array_size));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(size_t) * array_size*num_segments));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_out, sizeof(size_t) * array_size*num_segments));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_offsets, sizeof(int) * (num_segments+1)));
//  h_keys_out = new size_t[num_segments*array_size];
  
  for (i=0; i<=num_segments;i++) {
    h_offsets[i] = (int)(i*array_size);
  }

  CubDebugExit(cudaMemcpy(d_keys_in, arr, sizeof(size_t) * array_size * num_segments, cudaMemcpyHostToDevice));

  CubDebugExit(cudaMemcpy(d_offsets, h_offsets, sizeof(int) * num_segments, cudaMemcpyHostToDevice));

  size_t  temp_storage_bytes  = 0;
  void    *d_temp_storage     = NULL;

    // THIS or THAT
    //    CubDebugExit(         DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, array_size));
  CubDebugExit(DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, (int)array_size*num_segments, num_segments, d_offsets, &d_offsets[1]));

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    //    CubDebugExit(         DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, array_size));
  CubDebugExit(DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, (int)array_size*num_segments, num_segments, d_offsets, d_offsets + 1));

  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Maybe check that the example match what's in: https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#aac3b51925d55aa4504a29cb1379ae42e

  CubDebugExit(cudaMemcpy(arr, d_keys_out,sizeof(size_t) * array_size*num_segments, cudaMemcpyDeviceToHost));

#ifdef DEBUG
  printf("quickSortList: num_arrays, array_size = %ld, %ld\n",num_arrays,array_size);
#endif
  /*
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target map(to:num_arrays,array_size)
#pragma omp teams distribute parallel for schedule(static,1)
#else
#pragma omp parallel for shared(stack,arr) 
#endif
#endif
  for (i = 0; i < num_arrays; i++) {
    size_t *this_window = &arr[i*array_size];
    //    if (i%1000 == 0) printf("i = %zu\n",i);
    //    printf("i = %zu, this_window[array_size-1] = %zu\n",i,this_window[array_size]);
    //    unsigned long *this_stack = &stack[i*array_size];
//  quickSortIterative(this_window,this_stack,0,array_size-1);

//    unsigned long *this_window = &arr[i*array_size];
//    unsigned long   *h_keys             = new unsigned long[array_size];
//    unsigned long   *h_work             = new unsigned long[array_size];
    

//  cub::DoubleBuffer<unsigned long> d_keys(h_keys, h_work);

//  for (j=1; j < array_size; j++) {
//     d_keys.d_buffers[0][j] = d_keys[0]; 
//  }
    // Initialize device arrays
    // with segmented, gives out-of-bounds access error 700
    CubDebugExit(cudaMemcpy(d_keys_in, this_window, sizeof(size_t) * array_size, cudaMemcpyHostToDevice));
    // THIS ABOVE LINE IS NOT IN THE SNIPPET OF https://nvlabs.github.io/cub/structcub_1_1_device_segmented_radix_sort.html#a175a8f431517e609e3d683076de86402


                  // CUB segmented Radix Sort
    h_offsets[0] = 0;                                         // CUB segmented Radix Sort
    h_offsets[1] = (int)array_size;                      // CUB segmented Radix Sort

    CubDebugExit(cudaMemcpy(d_offsets, h_offsets, sizeof(int) * num_segments, cudaMemcpyHostToDevice));

    // Run

    // THIS or THAT
    //    CubDebugExit(         DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, array_size));
  // Allocate temporary storage
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;

    // THIS or THAT
    //    CubDebugExit(         DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, array_size));
    CubDebugExit(DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, array_size, num_segments, d_offsets, &d_offsets[1]));

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    //    CubDebugExit(         DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, array_size));
    CubDebugExit(DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,d_keys_out, array_size, num_segments, d_offsets, d_offsets + 1));

    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Maybe check that the example match what's in: https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#aac3b51925d55aa4504a29cb1379ae42e

    CubDebugExit(cudaMemcpy(this_window, d_keys_out,sizeof(unsigned long) * array_size, cudaMemcpyDeviceToHost));
  }
  */
  //  if (h_keys) delete[] h_keys;
    // Cleanup
  if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
  if (d_keys_out) CubDebugExit(g_allocator.DeviceFree(d_keys_out));
  if (d_offsets) CubDebugExit(g_allocator.DeviceFree(d_offsets));
  if (h_offsets) delete[] h_offsets;



//////////    return 0;

}


