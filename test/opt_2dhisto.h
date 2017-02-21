#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins, uint32_t* g_bins);

/* Include below the function headers of any other functions that you implement */

void* AllocateDevice(size_t size);

void CopyToDevice(void* D_device, void* D_host, size_t size);

void CopyFromDevice(void* D_host, void* D_device, size_t size);

void FreeDevice(void* D_device);

#endif
