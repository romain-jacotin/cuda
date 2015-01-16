# CUDA Programming

The goal of this document is only to make it easier for new developpers to undestand the overall CUDA paradigm and NVIDIA GPU features.  

I write this document as a global introduction in mind (the big picture) before installing the CUDA Toolkit, reading the CUDA C Programming guide, CUDA Runtime API and code samples.  

**For a complete and deep overview of the CUDA Toolkit and CUDA Runtime API and CUDA Libraries, please consult NVIDIA websites:**

* The CUDA Zone:
    * [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)  
* Join The CUDA Registered Developer Program:
    * [https://developer.nvidia.com/registered-developer-program](https://developer.nvidia.com/registered-developer-program)  
* Information about the CUDA Toolkit:
    * [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)  
* Download the latest CUDA Toolkit (Mac OSX / Linux x86 / Linux ARM / Windows):
    * [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)  

## Table of contents

* [Introduction](#introduction)
* [Thread Hierarchy](#threadhierarchy)
    * [Thread](#thread)
    * [Warp](#warp)
    * [Block](#block)
    * [Grid](#grid)
    * [Kernel](#kernel)
* [Memory](#memory)
    * [Host Memory](#hostmemory)
        * [Page-Locked Host Memory](#pagelockedhostmemory)
    * [Device Memory](#devicememory)
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
    * [Peer-to-peer](#peertopeer)
* [Versioning and Compatibility](#versioningandcompatibility)

## <A name="introduction"></A>  Introduction

![CPU versus GPU design](./images/cpu_vs_gpu.png "CPU versus GPU design")

GPU is specialized for compute-intensive, highly parallel computation - exactly what graphics rendering is about - and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control.

The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.

A multiprocessor is designed to execute hundreds of threads concurrently. To manage such a large amount of threads, it employs a unique architecture called SIMT (Single-Instruction, Multiple-Thread). The instructions are pipelined to leverage instruction-level parallelism within a single thread, as well as thread-level parallelism extensively through simultaneous hardware multithreading. Unlike CPU cores they are issued in order however and there is no branch prediction and no speculative execution.

## <A name="threadhierarchy"></A> Thread Hierarchy
### <A name="thread"></A> Thread
### <A name="warp"></A> Warp

The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. The term warp originates from weaving, the first parallel thread technology.

### <A name="block"></A> Block
### <A name="grid"></A> Grid
### <A name="kernel"></A> Kernel

## <A name="memory"></A> Memory

The GPU card contains its own DRAM memory separatly from the CPU's DRAM. Let's see how these memory works and can exchange information between them.

### <A name="hostmemory"></A> Host Memory
#### <A name="pagelockedhostmemory"></A> Page-Locked Host Memory
### <A name="devicememory"></A> Device Memory
#### <A name="sharedmemory"></A> Shared Memory
#### <A name="globalmemory"></A> Global Memory
### <A name="unifiedmemory"></A>Unified Memory

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

## <A name="multidevicesystem"></A> Multi-Device System
### <A name="peertopeer"></A> Peer-to-Peer

## <A name="versioningandcompatibility"></A> Versioning and Compatibility

![Versioning and Compatibility](./images/versioning_and_compatibility.png "Versioning and Compatibility")

* __The driver API is backward compatible__, meaning that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases.

* __The driver API is not forward compatible__, which means that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver.

* Only one version of the CUDA device driver can be installed on a system.

* All plug-ins and libraries used by an application must use the same version of:
    * any CUDA libraries (such as cuFFT, cuBLAS, ...)
    * the associated CUDA runtime.
