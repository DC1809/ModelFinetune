## Overview

This project explores three system-level operations on small language models: architecture parsing, efficient fine-tuning, and simple model composition.

For **Task 1**, I built a reusable parser that inspects each model at two levels: the raw PyTorch module tree and a normalized semantic summary. I used it on **GPT-2** and **TinyLlama** to recover a clean hierarchical representation of embeddings, repeated decoder blocks, attention modules, MLPs, normalization layers, and output heads.

For **Task 2**, I fine-tuned **TinyLlama/TinyLlama-1.1B-Chat-v1.0** using **PEFT/LoRA** on a compact **IMDB sentiment** dataset in a prompt-completion format.

For **Task 3**, I merged the trained LoRA adapter back into the base model and compared the **base model**, **adapter-loaded model**, and **merged model**.

## Key Results

- Base TinyLlama generation accuracy: **0.72**
- LoRA-adapted model generation accuracy: **0.93**
- Merged model generation accuracy: **0.94**
- Adapter vs merged prediction match rate: **0.97**
- Trainable parameters: **4,505,600 / 1,104,553,984** (~**0.408%**)
- Training runtime: about **1.17 minutes**

## Design Decisions

I structured the parser in two layers.

The first layer is a **raw recursive parser** that walks the `nn.Module` tree and records module names, module types, parameter counts, and nesting depth. This gives an exact hierarchical representation of the implementation, similar to parsing a document tree.

The second layer is a **normalized semantic parser** that maps model-specific modules into shared categories such as embeddings, decoder blocks, attention projections, MLP projections, norms, and LM heads. I chose this because raw traversal is faithful but hard to compare across architectures, while normalization makes GPT-2 and TinyLlama comparable through a common schema.

The main trade-off was between **generality** and **clarity**. A purely generic parser would be simpler, but less useful for downstream reasoning. A fully architecture-specific parser would be more precise, but harder to reuse. My final design keeps the raw tree intact while adding a lightweight interpretation layer on top.

A key design decision was to make the parser useful beyond visualization. I reused the parsed architecture summary to infer LoRA target modules for TinyLlama (`q_proj`, `k_proj`, `v_proj`, `o_proj`), which connects Task 1 directly to Task 2.

## Working Conditions

I worked in **Google Colab** on a hosted GPU runtime.

- GPU: **NVIDIA RTX PRO 6000 Blackwell Server Edition**
- GPU memory: **95 GB**
- Main libraries: **PyTorch, Transformers, PEFT, Datasets, Matplotlib, Pandas**

Approximate runtimes:
- Task 1 parsing and visualization: short interactive runtime in Colab
- Task 2 fine-tuning: about **1.17 minutes**
- Task 3 merge and comparison: short additional runtime

The main bottleneck was not the model logic itself, but **notebook/package stability**. I ran into package compatibility issues and had to simplify the stack and keep the training pipeline minimal. On the modeling side, the main constraint was efficiency: TinyLlama is small enough for Colab, but still large enough that full fine-tuning would be wasteful. Using LoRA with a compact dataset slice was the right trade-off.

## Extensibility and Scalability

To scale this system from 2 models to 10 or 50, I would formalize the parser output into a stable schema with fields such as model family, block container, repeated block type, norm style, attention projection style, MLP projection style, and candidate adaptation targets.

At that point, the workflow becomes:

1. parse model  
2. normalize structure  
3. evaluate on a standard suite  
4. choose adaptation strategy  
5. fine-tune  
6. merge or retain adapter  
7. report results

The first thing that would break at scale is not recursive traversal. It would be the **semantic classification layer**. Different architectures expose different naming conventions, projection layouts, positional encodings, and normalization styles. The raw tree scales easily; the fragile part is interpreting it correctly across families.

To handle models from different architectures such as **Llama vs Qwen**, I would keep one shared parser framework but add **family-specific normalization plugins**. The raw traversal stays universal, while the semantic layer becomes modular and architecture-aware.

## Creativity and Future Vision

If I were evolving this into a system that automatically improves models over time, I would build a loop like:

**parse → evaluate → detect weakness → train targeted adapter → merge selectively → re-evaluate**

My top three ideas would be:

**1. Architecture-aware adapter planning**  
Use the parser to choose LoRA targets automatically based on model structure rather than using one fixed target list for every family.

**2. Evaluation-driven adapter bank**  
Instead of always merging immediately, maintain a bank of small specialized adapters for different capability gaps such as sentiment, summarization, or instruction-following, and only activate or merge the ones that help.

**3. Safe merge gating**  
Do not treat merge as automatically good. Merge only when the improvement is stable on the target task and does not hurt a small regression set of general prompts.

The most overlooked improvement is turning the parser into a **control plane**, not just an inspection tool. Most people would stop at printing the model tree. I think its bigger value is operational: it can influence adaptation, evaluation, and merge strategy.

## Reflection

The most straightforward part of the project was building the first version of the recursive parser. PyTorch models are naturally hierarchical, so recovering the raw tree was conceptually simple.

The more challenging part was making the parser **meaningful across architectures**. GPT-2 and TinyLlama share a high-level decoder-only pattern, but their internal naming conventions and projection layouts differ significantly. That made semantic normalization and cross-model comparison more interesting than raw parsing.

The fine-tuning and merge workflow itself was conceptually straightforward, but practical notebook stability was more challenging than expected. A lot of systems work is not just about model code; it is also about making the pipeline reproducible and robust under hardware and package constraints.

I used external help for documentation, package troubleshooting, and implementation details around LoRA and merging. The final structure of the solution, especially the idea of reusing the parser output for LoRA target selection and future automation, is what made the project feel like a systems project rather than three disconnected tasks.
