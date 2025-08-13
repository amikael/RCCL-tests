# Testing the FINE_GRAIN Setting (`test_fine_grain2.sh`)

Fine-grained memory access (on AMD ROCm platforms) controls how tightly CPU and GPU share memory semantics when using PCIe-based memory transfers. On ROCm and MI250X, Fine-grained PCIe memory means memory that's allocated such that the GPU can read/write directly to the buffer without explicit synchronization.

Fine-grained memory access supports coherent access, i.e., CPU-GPU visibility of changes without explicit flush or copy. It is useful in communication (e.g., RCCL, NCCL, or zero-copy ops). This is different from coarse-grained memory, which requires synchronization barriers or manual flushes to ensure visibility between CPU and GPU.

The script `test_fine_grain2.sh` should tell whether fine-grained memory behaves as expected, although it doesn’t directly detect its status. It should help to test end-to-end visibility between CPU and GPU, and see where synchronization occurs. The correctness check (456 vs. 123) confirms memory visibility across PCIe.  However, when testing with the setting `HSA_FORCE_FINE_GRAIN_PCIE=0`, I could not see any contrast. Instead, the feature seemed to be on all the time:

=== Fine-Grained PCIe Memory Test ===
CUDA available: True
[HOST] Initial pinned memory value: 123
[DEVICE] Allocated empty tensor on GPU.
[COPY] Copying pinned host → device tensor.
[DEVICE] Value after copy: 123
[DEVICE] Writing new value 456 to device tensor.
[COPY] Copying device → pinned host tensor.
[HOST] Final pinned memory value (should be 456): 456
=== Done ===

Currently, I am not certain if the test gives false positives or works correctly.

