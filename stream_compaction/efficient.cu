#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        constexpr int BLOCK_SIZE = 256;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweepKernel(int n, int offset, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int rightIndex = (index + 1) * offset * 2 - 1;
            if (rightIndex < n) data[rightIndex] += data[rightIndex - offset];
        }

        __global__ void downSweepKernel(int n, int offset, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int rightIndex = (index + 1) * offset * 2 - 1;
            if (rightIndex < n) {
                int leftIndex = rightIndex - offset;
                int temp = data[leftIndex];
                data[leftIndex] = data[rightIndex];
                data[rightIndex] += temp;
            }
        }

        static void scanDevice(int n, int* data) {
            for (int offset = 1; offset < n; offset *= 2) {
                int activeThreads = n / (2 * offset);
                int blockCount = (activeThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                upSweepKernel << <blockCount, BLOCK_SIZE >> > (n, offset, data);
            }
            cudaMemset(data + n - 1, 0, sizeof(int));
            for (int offset = n / 2; offset >= 1; offset /= 2) {
                int activeThreads = n / (2 * offset);
                int blockCount = (activeThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                downSweepKernel << <blockCount, BLOCK_SIZE >> > (n, offset, data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int paddedSize = 1 << ilog2ceil(n);
            int* devData;
            cudaMalloc((void**)&devData, paddedSize * sizeof(int));
            cudaMemset(devData, 0, paddedSize * sizeof(int));
            cudaMemcpy(devData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            scanDevice(paddedSize, devData);
            timer().endGpuTimer();
            cudaMemcpy(odata, devData, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(devData);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            int paddedSize = 1 << ilog2ceil(n);
            int blockCount = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int* devInput;
            int* devOutput;
            int* devBools;
            int* devIndices;
            cudaMalloc((void**)&devInput, n * sizeof(int));
            cudaMalloc((void**)&devOutput, n * sizeof(int));
            cudaMalloc((void**)&devBools, paddedSize * sizeof(int));
            cudaMalloc((void**)&devIndices, paddedSize * sizeof(int));
            cudaMemset(devBools, 0, paddedSize * sizeof(int));
            cudaMemcpy(devInput, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean << <blockCount, BLOCK_SIZE >> > (n, devBools, devInput);
            cudaMemcpy(devIndices, devBools, paddedSize * sizeof(int), cudaMemcpyDeviceToDevice);
            scanDevice(paddedSize, devIndices);
            StreamCompaction::Common::kernScatter << <blockCount, BLOCK_SIZE >> > (n, devOutput, devInput, devBools, devIndices);
            timer().endGpuTimer();
            int lastIndex;
            cudaMemcpy(&lastIndex, devIndices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + (idata[n - 1] != 0);
            cudaMemcpy(odata, devOutput, count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(devInput);
            cudaFree(devOutput);
            cudaFree(devBools);
            cudaFree(devIndices);
            return count;
        }
    }
}
