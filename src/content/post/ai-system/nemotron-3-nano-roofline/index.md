---
title: "Experiment: Drawing Roofline for Nemotron-3 Nano"
description: "This post introduce experiment of drawing roofline for Nemotron-3 Nano model with Nsight Compute and vLLM."
publishDate: "8 April 2026"
updatedDate: "8 April 2026"
tags: ["AI_SYSTEMS", "PROFILING"]
---

## Purpose
The purpose of this experiment is as follows:
1. Analyze the compute and memory overhead of MLP Layer, Attention Layer, and Mamba2 Layer as sequence length increases.(I fixed the batch size to 1 for this experiment)
2. Draw roofline for MLP Layer, Attention Layer, and Mamba2 Layer as sequence length increases.

## Tools
1. Nsight Compute: This is a profiling tool provided by NVIDIA. It allows us to analyze the detailed performance of each kernels, and also provides various metrics such as achieved occupancy, achieved FLOPS, and achieved memory bandwidth.
2. Nsight Systems: This is another profiling tool provided by NVIDIA. It allows us to analyze the overall performance of the application, and also provides various metrics such as CPU utilization, GPU utilization, and memory usage.
3. vLLM: State of the art inference engine for LLM.

## Experiment Settings

### Hardware
1. NVIDIA 3090 GPU 

### Software
1. Model: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
2. CUDA version: 12.8
3. NVIDIA Driver version: 570.133.07
4. vLLM version: 0.18.0
5. nsight compute version: 2024.1.1.0 
6. nsight systems version: 2026.2.1.210

### Profile Space
1. Prompt Length: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
2. Batch Size: 1

## Experiment Implementation Details
1. **Eager mode and offline inference**: I used eager mode vLLM offline inference API to run the inference of Nemotron-3 Nano model.
2. **Chunked prefill with chunk size of 4096**: I used chunked prefill with chunk size of 4096. This means if prefill sequence length is less than 4096, I will run prefill with the actual sequence length. However, if prefill sequence length is greater than 4096, I will run prefill with chunk size of 4096 until I reach the actual sequence length. For example, if prefill sequence length is 5000, I will run prefill with chunk size of 4096 for the first 4096 tokens, and then run prefill with chunk size of 904 for the remaining tokens.
3. **Monkey patching**: I used monkey patching to insert nvtx marker for each layer, and also to insert "cudaProfileStart" and "cudaProfileStop" for each layer. This allows me to analyze the performance of each layer separately in Nsight Compute and Nsight Systems.
4. **Prefill the prompt and decode 1 token**: I setup the experiment to prefill the prompt and decode 1 token. This is because I want to analyze the performance of each layer during prefill, and also during decode. For example, if the prompt length is 8192, then vllm will process as follows: "prefill 4096 tokens" -> "prefill 4096 tokens"  -> "decode 1 token". By doing so, I can analyze the performance of each layer during prefill and decode separately.

## Experiment Code
[Experiment Code](https://github.com/jinho-choi123/roofline_nemotron_3_nano)

## Experiment Results
Comming soon!