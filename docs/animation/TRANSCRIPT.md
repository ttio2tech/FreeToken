# FreeToken engine animation - voiceover transcript

Render: `manim -qh freetoken_engine.py S0_Title S1_Problem S2_Pipeline S3_ExpertCache S4_Hybrid S5_PrefillStream S6_Memory S7_SemanticCache`
(then concatenate in order). Each block below is the narration for the scene of the same name.

## S0_Title (~6 s)
FreeToken is a serving engine built for one job: running frontier-scale mixture-of-experts models on the hardware you already own. Let's walk through how it works.

## S1_Problem (~25 s)
A 290-billion-parameter model needs well over a hundred gigabytes of weights. A gaming GPU has twenty-four. It simply does not fit.
But a mixture-of-experts model does not use all of its weights for every token. In each layer a router picks only a handful of experts, here six out of sixty-four.
So FreeToken keeps every weight in host RAM and keeps only the experts that are hot on the GPU.

## S2_Pipeline (~30 s)
A request enters through an OpenAI or Anthropic compatible HTTP API, goes to a tokenizer, and reaches the scheduler. These run as separate processes joined by message queues, so text processing never stalls the GPU.
The scheduler first matches the prompt against a radix prefix cache. Tokens it has already computed are reused, not recomputed.
The rest of the prompt is prefilled in chunks, which bounds memory and lets other requests make progress.
Once prefill is done, every running request joins one decode batch, producing one token per step. The engine runs the model and the detokenizer streams text back.

## S3_ExpertCache (~35 s)
During decode, expert weights live in pinned host memory. On the GPU there is a pool of slots, managed as an LRU cache.
At every layer, the router output goes through ensure-experts. Experts already in a slot are hits and cost nothing. Missing experts are given a slot, evicting the least recently used one, and are copied across PCIe.
Over successive steps the same experts keep getting picked, so the hit rate climbs. Routing skew is what makes this cache pay off.

## S4_Hybrid (~40 s)
Fetching every miss over PCIe makes the GPU wait, while the CPU sits idle next to fast system memory. Hybrid mode uses both.
Measured at startup, the CPU reads DRAM at around one hundred gigabytes per second, the GPU gets about twenty-five over PCIe. FreeToken fetches the fraction pcie over pcie plus cpu of the misses, and computes the rest on the CPU, so both sides finish at the same time.
With ten misses, two are fetched and eight are computed on the CPU. The partial outputs are merged and the next layer starts.
The hand-off uses GPU stream memory operations and flags rather than host callbacks, so it stays inside CUDA graphs.

## S5_PrefillStream (~30 s)
Prefill is different. A long prompt touches nearly every expert, so caching individual experts does not help.
Instead the engine streams whole layers through two GPU buffers. While the GPU computes layer i in one buffer, a copy stream is already loading layer i plus one into the other.
Experts that are already resident in the slot cache are gathered on the device, so only true misses cross PCIe.

## S6_Memory (~30 s)
GPU memory is one budget. After weights, FreeToken reserves a minimum of KV cache, gives expert slots as much as they can use, and hands what is left to KV pages.
That boundary is not fixed. Memory can move between the expert cache and the KV cache at runtime, with no restart and no weight reload.

## S7_SemanticCache (~35 s)
Agents edit their own context: they call a tool, drop a thinking block, append a result. A plain prefix cache would throw away everything after the first change.
For hybrid models with recurrent state, FreeToken checkpoints that state at semantic anchors, such as the opening of a tool call.
When the context changes, the engine restores the anchor and recomputes only the edited tail, instead of the entire conversation.
