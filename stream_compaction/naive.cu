#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        constexpr int BLOCK_SIZE = 256;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernel(int n, int offset, int* odata, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < n) {
                if (index >= offset) odata[index] = idata[index] + idata[index - offset];
                else odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* devInput;
            int* devOutput;
            cudaMalloc((void**)&devInput, n * sizeof(int));
            cudaMalloc((void**)&devOutput,  n * sizeof(int));
            cudaMemcpy(devInput, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int blockCount = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer().startGpuTimer();
            for (int offset = 1; offset < n; offset *= 2) {
                kernel<<<blockCount, BLOCK_SIZE >>>(n, offset, devOutput, devInput);
                int* temp = devInput;
                devInput = devOutput;
                devOutput = temp;
            }
            timer().endGpuTimer();
            odata[0] = 0;
            if (n > 1) cudaMemcpy(odata + 1, devInput, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(devInput);
            cudaFree(devOutput);
        }
    }
}
