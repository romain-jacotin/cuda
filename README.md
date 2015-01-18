# CUDA Programming

The goal of this document is only to make it easier for new developpers to undestand the overall CUDA paradigm and NVIDIA GPU features.  

This document is just a high level overview of CUDA features and CUDA programming model. It is probably simpler to read it before installing the CUDA Toolkit, reading the CUDA C Programming guide, CUDA Runtime API and code samples.  

**For a complete and deep overview of the CUDA Toolkit and CUDA Runtime API and CUDA Libraries, please consult NVIDIA websites:**

* The CUDA Zone:
    * [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)  
* Join The CUDA Registered Developer Program:
    * [https://developer.nvidia.com/registered-developer-program](https://developer.nvidia.com/registered-developer-program)  
* Information about the CUDA Toolkit:
    * [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)  
* Download the latest CUDA Toolkit (Mac OSX / Linux x86 / Linux ARM / Windows):
    * [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)  

----------------------------

## Table of contents

* [Introduction](#introduction)
    * [Terminology](#terminology)
    * [CUDA Software Architecture](#cudasoftwarearchitecture)
    * [Versioning and Compatibility](#versioningandcompatibility)
    * [CUDA Libraries](#parallellibraries)
        * [cuBLAS](#cublas)
        * [cuSPARSE](#cusparse)
        * [cuSOLVER](#cusolver)
        * [cuFFT](#cufft)
        * [cuRAND](#curand)
        * [cuDNN](#dnn)
        * [NPP](#npp)
* [Thread Hierarchy](#threadhierarchy)
    * [Kernel](#kernel)
    * [Thread](#thread)
    * [Warp](#warp)
    * [Block](#block)
    * [Grid](#grid)
* [Memory](#memory)
    * [Host Memory](#hostmemory)
    * [Device Memory](#devicememory)
    * [Unified Memory](#unifiedmemory)
* [Asynchronous Concurrency Execution](#asynchronousconcurrencyexecution)
    * [Concurrent Data Access](#Concurrent Data Access)
        * [Synchronize Functions](#synchronizefunctions)
        * [Atomic Functions](#atomicfunctions)
    * [Concurrent Kernel Execution](#concurrentkernelexecution)
    * [Concurrent Data Transfers](#concurrentdatatransfers)
    * [Streams](#streams)
        * [Callbacks](#callbacks)
        * [Events](#events)
    * [Dynamic Parallelism](#dynamicparallelism)
    * [Hyper-Q](#hyperq)
* [Multi-Device System](#multidevicesystem)
    * [Stream and Event Behavior](#streamandeventbehavior)
    * [GPU Direct](#gpudirect)

----------------------------

## <A name="introduction"></A>  Introduction

![CPU versus GPU design](./images/cpu_vs_gpu.png "CPU versus GPU design")

GPU is specialized for compute-intensive, highly parallel computation - exactly what graphics rendering is about - and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control.

The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.

A multiprocessor is designed to execute hundreds of threads concurrently. To manage such a large amount of threads, it employs a unique architecture called SIMT (Single-Instruction, Multiple-Thread). The instructions are pipelined to leverage instruction-level parallelism within a single thread, as well as thread-level parallelism extensively through simultaneous hardware multithreading. Unlike CPU cores they are issued in order however and there is no branch prediction and no speculative execution.

### <A name="terminology"></A> Terminology

* __HOST__: The CPU and its memory (Host Memory)

* __DEVICE__: The GPU and its memory (Device Memory)

* __COMPUTE CAPABILITY__:  The compute capability of a device is represented by a version number, also sometimes called its "SM version". This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.
    * The compute capability version comprises a major and a minor version number (x.y):
        * Devices with the same major revision number are of the same core architecture:
            * 1 = __Telsa__ architecture
            * 2 = __Fermi__ architecture
            * 3 = __Kepler__ architecture
            * 5 = __Maxwell__ architecture
        * The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new features.

Notes:

* _The compute capability version of a particular GPU should not be confused with the CUDA version (e.g., CUDA 5.5, CUDA 6, CUDA 6.5, CUDA 7.0), which is the version of the CUDA software platform._
* _The Tesla architecture (Compute Capability 1.x) is no longer supported starting with CUDA 7.0._

| Architecture specifications  Compute capability (version) | 1.0 | 1.1 | 1.2 | 1.3 | 2.0 | 2.1 | 3.0 | 3.5 | 5.0 | 5.2 |
|---|---|---|---|---|---|---|---|---|---|---| 
| _Number of cores for integer and floating-point arithmetic functions operations_ | 8 | 8 | 8 | 8 | 32 | 48 | 192 | 192 | 128 | 128 |

|   |   |
|---|---|
| ![MacBook Pro](./images/macbookpro.jpg "MacBook Pro") | MacBook Pro 15' <BR> NVIDIA GeForce GT 750M 2 Giga <BR><BR> __384 cores with 2 GB__ |
| ![Tesla K40](./images/tesla_k40.png "Tesla K40") | NVIDIA Tesla GPU Accelerator K40 <BR><BR> __2.880 cores with 12 GB__ |

### <A name="cudasoftwarearchitecture"></A> CUDA Software Architecture

![CUDA Software Architecture](./images/cuda_design.jpg "CUDA Software Architecture")

CUDA is composed of two APIs:

* A low-level API called the __CUDA Driver API__,
* A higher-level API called the __CUDA Runtime API__ that is implemented on top of the CUDA driver API.

These APIs are mutually exclusive: An application should use either one or the other.

* The CUDA runtime eases device code management by providing implicit initialization, context management, and module management.
* In contrast, the CUDA driver API requires more code, is harder to program and debug, but offers a better level of control and is language-independent since it only deals with cubin objects. In particular, it is more difficult to configure and launch kernels using the CUDA driver API, since the execution configuration and kernel parameters must be specified with explicit function calls. Also, device emulation does not work with the CUDA driver API.

The easiest way to write an application that use GPU acceleration is to use existing CUDA libraries if they correspond to your need: cuBLAS, cuRAND, cu FFT, cuSOLVER, cuDNN, ...

But if existing libraries does not fit your need then you must write your own GPU functions in C language ( CUDA call them kernels, and CUDA C program have ".cu" file extension ) and use them in your application by using CUDA Runtime API, and linking CUDA Runtime library and home made kernels functions to your application.

### <A name="versioningandcompatibility"></A> Versioning and Compatibility

![Versioning and Compatibility](./images/versioning_and_compatibility.png "Versioning and Compatibility")

* __The driver API is backward compatible__, meaning that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases.

* __The driver API is not forward compatible__, which means that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver.

* Only one version of the CUDA device driver can be installed on a system.

* All plug-ins and libraries used by an application must use the same version of:
    * any CUDA libraries (such as cuFFT, cuBLAS, ...)
    * the associated CUDA runtime.

### <A name"cudalibraries"></A> CUDA Libraries

#### <A name="cublas"></A> cuBLAS (Basic Linear Algebra Subroutines)

The NVIDIA CUDA Basic Linear Algebra Subroutines (cuBLAS) library is a GPU-accelerated version of the complete standard BLAS library.
      
* Complete support for all 152 standard BLAS routines
* Single, double, complex, and double complex data types
* Support for CUDA streams
* Fortran bindings
* Support for multiple GPUs and concurrent kernels
* Batched GEMM API
* Device API that can be called from CUDA kernels
* Batched LU factorization API
* Batched matrix inverse API
* New implementation of TRSV (Triangular solve), up to 7x faster than previous implementation.

![cublas](./images/cublas.jpg "cublas")

_To learn more about cuBLAS:_ [https://developer.nvidia.com/cuBLAS](https://developer.nvidia.com/cuBLAS)

#### <A name="cusparse"></A> cuSPARSE (Sparse Matrix)

The NVIDIA CUDA Sparse Matrix library (cuSPARSE) provides a collection of basic linear algebra subroutines used for sparse matrices that delivers up to 8x faster performance than the latest MKL. The latest release includes a sparse triangular solver.

Supports dense, COO, CSR, CSC, ELL/HYB and Blocked CSR sparse matrix formats
Level 1 routines for sparse vector x dense vector operations
Level 2 routines for sparse matrix x dense vector operations
Level 3 routines for sparse matrix x multiple dense vectors (tall matrix)
Routines for sparse matrix by sparse matrix addition and multiplication
Conversion routines that allow conversion between different matrix formats
Sparse Triangular Solve
Tri-diagonal solver
Incomplete factorization preconditioners ilu0 and ic0

![cusparse](./images/cusparse.jpg "cusparse")

_To learn more about cuSPARSE:_ [https://developer.nvidia.com/cuSPARSE](https://developer.nvidia.com/cuSPARSE)

#### <A name="cusolver"></A> cuSOLVER (Solver)

The NVIDIA cuSOLVER library provides a collection of dense and sparse direct solvers which deliver significant acceleration for Computer Vision, CFD, Computational Chemistry, and Linear Optimization applications.

* __cusolverDN__: Key LAPACK dense solvers 3-6x faster than MKL.
    * Dense Cholesky, LU, SVD, QR
    * Applications include: optimization, Computer Vision, CFD
* __cusolverSP__
    * Sparse direct solvers & Eigensolvers
    * Applications include: Newton's method, Chemical Kinetics
* __cusolverRF__
    * Sparse refactorization solver
    * Applications include: Chemistry, ODEs, Circuit simulation

_To learn more about cuSOLVER:_ [https://developer.nvidia.com/cuSOLVER](https://developer.nvidia.com/cuSOLVER)

#### <A name="cufft"></A> cuFFT (Fast Fourier Transformation)

The FFT is a divide-and-conquer algorithm for efficiently computing discrete Fourier transforms of complex or real-valued data sets. It is one of the most important and widely used numerical algorithms in computational physics and general signal processing.

![cufft](./images/cufft.jpg "cuFFT")

_To learn more about cuFFT:_ [https://developer.nvidia.com/cuFFT](https://developer.nvidia.com/cuFFT)

#### <A name="curand"></A> cuRAND (Random Number)

The cuRAND library provides facilities that focus on the simple and efficient generation of high-quality pseudorandom and quasirandom numbers. A pseudorandom sequence of numbers satisfies most of the statistical properties of a truly random sequence but is generated by a deterministic algorithm. A quasirandom sequence of n -dimensional points is generated by a deterministic algorithm designed to fill an n-dimensional space evenly.

Random numbers can be generated on the device or on the host CPU.

![cuRAND](./images/curand.jpg "cuRAND")

_To learn more about cuRAND:_ [https://developer.nvidia.com/cuRAND](https://developer.nvidia.com/cuRAND)

#### <A name="cudnn"></A> cuDNN (Deep Neural Network)

![cuDNN](./images/cudnn.png "cuDNN")

cuDNN is a GPU-accelerated library of primitives for deep neural networks. It provides highly tuned implementations of routines arising frequently in DNN applications:

* Convolution forward and backward, including cross-correlation
* Pooling forward and backward
* Softmax forward and backward
* Neuron activations forward and backward:
    * Rectified linear (ReLU)
    * Sigmoid
    * Hyperbolic tangent (TANH)
* Tensor transformation functions

_To learn more about cuDNN:_ [https://developer.nvidia.com/cuDNN](https://developer.nvidia.com/cuDNN)

#### <A name="npp"></A> NPP (NVIDIA Performance Primitive)

NVIDIA Performance Primitive (NPP) is a library of functions for performing CUDA accelerated processing. The initial set of
functionality in the library focuses on imaging and video processing and is widely applicable for developers
in these areas. NPP will evolve over time to encompass more of the compute heavy tasks in a variety of problem domains.

* __Eliminates unnecessary copying of data to/from CPU memory__
    * Process data that is already in GPU memory
    * Leave results in GPU memory so they are ready for subsequent processing
* __Data Exchange and Initialization__
    * Set, Convert, Copy, CopyConstBorder, Transpose, SwapChannels
* __Arithmetic and Logical Operations__
    * Add, Sub, Mul, Div, AbsDiff, Threshold, Compare
* __Color Conversion__
    * RGBToYCbCr, YcbCrToRGB, YCbCrToYCbCr, ColorTwist, LUT_Linear
* __Filter Functions__
    * FilterBox, Filter, FilterRow, FilterColumn, FilterMax, FilterMin, Dilate, Erode, SumWindowColumn, SumWindowRow
* __JPEG__
    * DCTQuantInv, DCTQuantFwd, QuantizationTableJPEG
* __Geometry Transforms__
    * Mirror, WarpAffine, WarpAffineBack, WarpAffineQuad, WarpPerspective, WarpPerspectiveBack  , WarpPerspectiveQuad, Resize
* __Statistics Functions__
    * Mean_StdDev, NormDiff, Sum, MinMax, HistogramEven, RectStdDev

_To learn more about NPP:_ [https://developer.nvidia.com/NPP](https://developer.nvidia.com/NPP)

----------------------------

## <A name="threadhierarchy"></A> Thread Hierarchy

CUDA C extends C by allowing the programmer to define C functions, called kernels, that, when called, are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions.

The CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C program. This is the case, for example, when the kernels execute on a GPU and the rest of the Golang program executes on a CPU.

Calling a kernel function from the Host launch a grid of thread blocks on the Device:

![Heterogeneous Programming](./images/heterogeneous_programming.png "Heterogeneous Programming")

### <A name="kernel"></A> Kernel

A kernel is a special C function defined using the `__global__` declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new <<<...>>>execution configuration syntax. Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through the built-in threadIdx variable.

```c
// Kernel definition __global__ void VecAdd(float* A, float* B, float* C) { int i = threadIdx.x; C[i] = A[i] + B[i]; } int main() { ... // Kernel invocation with N threads VecAdd<<<1, N>>>(A, B, C); ... } - See more at: file:///Developer/NVIDIA/CUDA-7.0/doc/html/cuda-c-programming-guide/index.html#page-locked-host-memory
```

![Thread Hierarchy](./images/grid_of_thread_blocks.png "Thread Hierarchy")

### <A name="thread"></A> Thread
### <A name="warp"></A> Warp

The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. The term warp originates from weaving, the first parallel thread technology.

![Thread Divergence](./images/thread_divergence.png "Thread Divergence")

### <A name="block"></A> Block
### <A name="grid"></A> Grid

----------------------------

| Technical Specifications per Compute Capability | 1.1 |  1.2 - 1.3 | 2.x | 3.0 - 3.2 | 3.5 | 3.7 | 5.0 | 5.2 |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| __Grid of thread blocks__ | | | | | | | | |
| _Maximum dimensionality of a grid_ | 2 | 2 | 3 | 3 | 3 | 3 | 3 | 3 |
| _Maximum x-dimension of a grid_ | 65535 | 65535 | 65535 | 2^31-1 | 2^31-1 | 2^31-1 | 2^31-1 | 2^31-1 |
| _Maximum y- or z-dimension of a grid_ | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 |
| | | | | | | | | |
| __Thread Block__ | | | | | | | | |
| _Maximum dimensionality of a block_ | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| _Maximum x- or y-dimension of a block_ | 512 | 512 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| _Maximum z-dimension of a block_ | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| _Maximum number of threads per block_ | 512 | 512 | 1024 | 1024 | 1 024 | 1.024 | 1 024 | 1024 |
| | | | | | | | | |
| __Per Multiprocessor__ | | | | | | | | |
| _Warp size_ | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| _Maximum number of resident blocks_ | 8 | 8 | 8 | 16 | 16 | 16 | 32 | 32 |
| _Maximum number of resident warps_ | 24 | 32 | 48 | 64 | 64 | 64 | 64 | 64 |
| _Maximum number of resident threads_ | 768 | 1024 | 1536 | 2048 | 2048 | 2048 | 2048 | 2048 |

----------------------------

## <A name="memory"></A> Memory

The GPU card contains its own DRAM memory separatly from the CPU's DRAM.

![CUDA Compute Job](./images/cuda_compute_job.png "CUDA Compute Job")

1. A typical CUDA compute job begins by copying data to the GPU, usually to global memory.
2. The GPU then asynchronously runs the compute task it has been assigned.
3. When the host makes a call to copy memory back to host memory, the call will block until all threads have completed and the data is available, at which point results are sent back.

### <A name="hostmemory"></A> Host Memory

* __Page-Locked Host Memory (Pinned Memory)__

The CUDA runtime provides functions to allow the use of page-locked (also known as pinned) host memory (as opposed to regular pageable host memory):

Using page-locked host memory has several benefits:

* Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices.
* On some devices, page-locked host memory can be mapped into the address space of the device, eliminating the need to copy it to or from device memory.
* On systems with a front-side bus, bandwidth between host memory and device memory is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as write-combining.

![Pinned Memory](./images/pinned_memory.jpg "Pinned Memory")

### <A name="devicememory"></A> Device Memory

CUDA threads may access data from multiple memory spaces during their execution:

* __Local memory__
    * Each thread has private local memory (__per-thread local memory__).
    * Local memory is so named because its scope is local to the thread, not because of its physical location. In fact, local memory is off-chip (=DRAM). Hence, access to local memory is as expensive as access to global memory. In other words, the term local in the name does not imply faster access.
* __Shared memory__
    * Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block (__per-block shared memory__).
    * Because it is on-chip, shared memory has much higher bandwidth and lower latency than local and global memory.
* __Global memory__
    * All threads have access to the same global memory (=DRAM).

![Memory Hierarchy](./images/memory_hierarchy.png "Memory Hierarchy")

----------------------------

| Technical Specifications per Compute Capability | 1.1 |  1.2 - 1.3 | 2.x | 3.0 - 3.2 | 3.5 | 3.7 | 5.0 | 5.2 |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| __Registers__ | | | | | | | | |
| _Number of 32-bit registers per multiprocessor_ | 8k | 16k | 32k | 64k | 64k | 128k | 64k | 64k |
| _Maximum number of 32-bit registers per thread block_ | | | 32k | 64k | 64k | 64k | 64k | 64k |
| _Maximum number of 32-bit registers per thread_ | 128 | 128 | 63 | 63 | 255 | 255 | 255 | 255 |
| | | | | | | | | |
| __Local Memory__ | | | | | | | | |
| _Maximum amount of local memory per thread_ | 16 KB | 16 KB | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB |
| | | | | | | | | |
| __Shared Memory__ | | | | | | | | |
| _Maximum amount of shared memory per multiprocessor_ | 16 KB | 16 KB | 48 KB | 48 KB | 48 KB | 112 KB | 64 KB | 96 KB |
| _Maximum amount of shared memory per thread block_   | 16 KB | 16 KB | 48 KB | 48 KB | 48 KB | 48 KB  | 48 KB | 48 KB |

### <A name="unifiedmemory"></A>Unified Memory

----------------------------

## <A name="asynchronousconcurrencyexecution"></A> Asynchronous Concurrency Execution
### <A name="concurrentdataaccess"></A> Concurrent Data Access
#### <A name="synchronizefunctions"></A> Synchronize Functions
#### <A name="atomicfunction"></A> Atomic Functions
### <A name="concurrentkernelexecution"></A> Concurrent kernel execution
### <A name="concurrentdatatransfers"></A> Concurrent Data Transfers
### <A name="streams"></A> Streams
#### <A name="callbacks"></A> Callbacks
#### <A name="events"></A> Events
### <A name="dynamicparallelism"></A> Dynamic Parallelism

Dynamic Parallelism enables a CUDA kernel to create and synchronize new nested work, using the CUDA runtime API to launch other kernels, optionally
synchronize on kernel completion, perform device memory management, and create and use streams and events, all without CPU involvement. 

__Note: Need Compute Capability >= 3.5__

![Dynamic Parallelism](./images/dynamic_parallelism.jpg "Dynamic Parallelism")

Dynamic Parallelism dynamically spawns new threads by adapting to the data without going back to the CPU, greatly simplifying GPU programming and accelerating  algorithms.

![Dynamic Parallelism](./images/dynamic_parallelism_example.jpg "Dynamic Parallelism")

### <A name="hyperq"></A> Hyper-Q

Hyper-Q enables multiple CPU threads or processes to launch work on a single GPU simultaneously,
thereby dramatically increasing GPU utilization and slashing CPU idle times.
This feature increases the total number of “connections” between the host and GPU by
allowing 32 simultaneous, hardware-managed connections, compared to the single
connection available with GPUs without Hyper-Q.
Hyper-Q is a flexible solution that allows connections for both CUDA streams and Message
Passing Interface (MPI) processes, or even threads from within a process. Existing
applications that were previously limited by false dependencies can see a dramatic
performance increase without changing any existing code.

__Note: Need Compute Capability >= 3.5__

![Hyper-Q](./images/hyperq.png "Hyper-Q")

----------------------------

## <A name="multidevicesystem"></A> Multi-Device System

A host system can have multiple GPU Devices.

### <A name="streamandeventbehavior"></A> Stream and Event Behavior

* A kernel launch will fail if it is issued to a stream that is not associated to the current device.
* A memory copy will succeed even if it is issued to a stream that is not associated to the current device.
* Each device has its own default stream, so commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device.

### <A name="gpudirect"></A> GPU Direct

**Accelerated communication with network and storage devices**  
Network and GPU device drivers can share “pinned” (page-locked) buffers, eliminating the need to make a redundant copy in CUDA host memory.

![RDMA](./images/rdma.png "RDMA")

**Peer-to-Peer Transfers between GPUs**  
Use high-speed DMA transfers to copy data between the memories of two GPUs on the same system/PCIe bus.

**Peer-to-Peer memory access**  
Optimize communication between GPUs using NUMA-style access to memory on other GPUs from within CUDA kernels.

![Peer-to-Peer](./images/peer_to_peer.png "Peer-to-Peer")

**RDMA**  

Eliminate CPU bandwidth and latency bottlenecks using remote direct memory access (RDMA) transfers between GPUs and other PCIe devices, resulting in significantly improved MPISendRecv efficiency between GPUs and other nodes)

**GPUDirect for Video**  
Optimized pipeline for frame-based devices such as frame grabbers, video switchers, HD-SDI capture, and CameraLink devices.

* GPUDirect version 1 supported accelerated communication with network and storage devices. It was supported by InfiniBand solutions available from Mellanox and others.
* GPUDirect version 2 added support for peer-to-peer communication between GPUs on the same shared memory server.
* GPU Direct RDMA enables RDMA transfers across an Infiniband network between GPUs in different cluster nodes, bypassing CPU host memory altogether. 

Using GPUDirect, 3rd party network adapters, solid-state drives (SSDs) and other devices can directly read and write CUDA host and device memory. GPUDirect eliminates unnecessary system memory copies, dramatically lowers CPU overhead, and reduces latency, resulting in significant performance improvements in data transfer times for applications running on NVIDIA products.

----------------------------
