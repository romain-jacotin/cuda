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
* [Thread Hierarchy](#threadhierarchy)
    * [Kernel](#kernel)
    * [Thread](#thread)
    * [Warp](#warp)
    * [Block](#block)
    * [Grid](#grid)
* [Memory](#memory)
    * [Host Memory](#hostmemory)
        * [Page-Locked Host Memory](#pagelockedhostmemory)
    * [Device Memory](#devicememory)
        * [Local Memory](#localmemory)
        * [Shared Memory](#sharedmemory)
        * [Global Memory](#globalmemory)
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
* [Multi-Device System](#multidevicesystem)
    * [Stream and Event Behavior](#streamandeventbehavior)
    * [Peer-to-peer](#peertopeer)
* [Versioning and Compatibility](#versioningandcompatibility)

----------------------------

## <A name="introduction"></A>  Introduction

![CPU versus GPU design](./images/cpu_vs_gpu.png "CPU versus GPU design")

GPU is specialized for compute-intensive, highly parallel computation - exactly what graphics rendering is about - and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control.

The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.

A multiprocessor is designed to execute hundreds of threads concurrently. To manage such a large amount of threads, it employs a unique architecture called SIMT (Single-Instruction, Multiple-Thread). The instructions are pipelined to leverage instruction-level parallelism within a single thread, as well as thread-level parallelism extensively through simultaneous hardware multithreading. Unlike CPU cores they are issued in order however and there is no branch prediction and no speculative execution.

### <A name="terminology"></A> Terminology

* __HOST__: The CPU and its memory (Host Memory)
* __DEVICE__: The GPU and its memory (Device Memory)

----------------------------

## <A name="threadhierarchy"></A> Thread Hierarchy

CUDA C extends C by allowing the programmer to define C functions, called kernels, that, when called, are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions.

The CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C program. This is the case, for example, when the kernels execute on a GPU and the rest of the Golang program executes on a CPU.

Calling a kernel function from the Host launch a grid of thread blocks on the Device:

![Heterogeneous Programming](./images/heterogeneous_programming.png "Heterogeneous Programming")

### <A name="kernel"></A> Kernel

![Thread Hierarchy](./images/grid_of_thread_blocks.png "Thread Hierarchy")

### <A name="thread"></A> Thread
### <A name="warp"></A> Warp

The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. The term warp originates from weaving, the first parallel thread technology.

### <A name="block"></A> Block
### <A name="grid"></A> Grid

----------------------------

## <A name="memory"></A> Memory

The GPU card contains its own DRAM memory separatly from the CPU's DRAM. Let's see how these memory works and can exchange information between them.

### <A name="hostmemory"></A> Host Memory
#### <A name="pagelockedhostmemory"></A> Page-Locked Host Memory
### <A name="devicememory"></A> Device Memory

There is 3 memory types inside the GPU Device:

* Local memory
* Shared memory
* Global memory

![Memory Hierarchy](./images/memory_hierarchy.png "Memory Hierarchy")

#### <A name="localmemory"></A> Local Memory

Each thread owns its personal **local memory**. It is stored for kernel's parameters and kernel variables associated with the thread.

#### <A name="sharedmemory"></A> Shared Memory

#### <A name="globalmemory"></A> Global Memory

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

Need Compute Capability >= 3.5

----------------------------

## <A name="multidevicesystem"></A> Multi-Device System

A host system can have multiple GPU Devices.

### <A name="streamandeventbehavior"></A> Stream and Event Behavior

* A kernel launch will fail if it is issued to a stream that is not associated to the current device.
* A memory copy will succeed even if it is issued to a stream that is not associated to the current device.
* Each device has its own default stream, so commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device.

### <A name="peertopeer"></A> Peer-to-Peer

![Peer-to-Peer](./images/peer_to_peer.png "Peer-to-Peer")

#### Peer-to-Peer Memory Access

* When the application is run as a 64-bit process, devices of compute capability 2.0 and higher from the Tesla series may address each other's memory (i.e., a kernel executing on one device can dereference a pointer to the memory of the other device).
* A unified address space is used for both devices (see Unified Virtual Address Space), so the same pointer can be used to address memory from both devices.

#### Peer-to-Peer Memory Copy

* Memory copies can be performed between the memories of two different devices.
* Consistent with the normal behavior of streams, an asynchronous copy between the memories of two devices may overlap with copies or kernels in another stream.

----------------------------

## <A name="versioningandcompatibility"></A> Versioning and Compatibility

![Versioning and Compatibility](./images/versioning_and_compatibility.png "Versioning and Compatibility")

* __The driver API is backward compatible__, meaning that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases.

* __The driver API is not forward compatible__, which means that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver.

* Only one version of the CUDA device driver can be installed on a system.

* All plug-ins and libraries used by an application must use the same version of:
    * any CUDA libraries (such as cuFFT, cuBLAS, ...)
    * the associated CUDA runtime.
