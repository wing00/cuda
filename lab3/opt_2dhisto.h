#ifndef OPT_KERNEL
#define OPT_KERNEL


void opt_2dhisto(uint32_t *deviceImage, uint8_t *deviceBins, size_t height, size_t width);

uint32_t *AllocateDeviceImage(size_t height, size_t width);
uint8_t *AllocateDeviceBins(size_t height, size_t width);

void ToDeviceImage(uint32_t *deviceImage, uint32_t **hostImage,  size_t height, size_t width);
void ToDeviceBins(uint8_t *deviceBins, uint8_t *hostBins, size_t height, size_t width);

void FromDeviceImage(uint32_t *hostImage, uint32_t *deviceImage, size_t height, size_t width);
void FromDeviceBins(uint8_t *hostBins, uint8_t *deviceBins, size_t height, size_t width);

void FreeDeviceImage(uint32_t *deviceImage);
void FreeDeviceBins(uint8_t *deviceBins);


#endif
