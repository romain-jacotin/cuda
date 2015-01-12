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
    * [SIMT](#simt)
* [Memory](#memory)
    * [Host Memory](#hostmemory)
    * [Device Memory](#devicememory)
    * [Shared Memory](#sharedmemory)
    * [Unified Memory](#unifiedmemory)
* [Thread Hierarchy](#threadhierarchy)
    * [Thread](#thread)
    * [Warp](#warp)
    * [Block](#block)
    * [Grid](#grid)
    * [Kernel](#kernel)
* [Concurrency Execution](#concurrencyexecution)
    * [Concurrent Kernel Execution](#concurrentkernelexecution)
    * [Concurrent Data Transfers](#concurrentdatatransfers)
    * [Streal](#stream)
    * [Callback](#callback)
    * [Event](#event)
    * [Dynamic Parallelism](#dynamicparallelism)
* [Multi-Device System](#multidevicesystem)
    * [Peer-to-peer](#peertopeer)

## <A name="introduction"></A>  Introduction
### <A name="simt"></A> SIMT

## <A name="memory"></A> Memory
### <A name="hostmemory"></A> Host Memory
### <A name="devicememory"></A> Device Memory
### <A name="sharedmemory"></A> Shared Memory
### <A name="unifiedmemory"></A>Unified Memory

## <A name="threadhierarchy"></A> Thread Hierarchy
### <A name="thread"></A> Thread
### <A name="warp"></A> Warp
### <A name="block"></A> Block
### <A name="grid"></A> Grid
### <A name="kernel"></A> Kernel

## <A name="concurrencyexecution"></A> Concurrency Execution
### <A name="concurrentkernelexecution"></A> Concurrent kernel execution
### <A name="concurrentdatatransfers"></A> Concurrent Data Transfers
### <A name="stream"></A> Stream
### <A name="callback"></A> Callback
### <A name="event"></A> Event
### <A name="dynamicparallelism"></A> Dynamic Parallelism

Need Compute Capability >= 3.5

## <A name="multidevicesystem"></A> Multi-Device System
### <A name="peertopeer"></A> Peer-to-Peer

