# A QUICK ENVIRONMENT CHECK

This simple command (also in `envcheck.sh`) is super useful:
```
env | egrep '^(RCCL|FI_|NCCL_|PLUGIN|LD_LIB|HSA_|WITH_CONDA|CXI_|SLURM_|MIOPEN_)' | sort
```
For example, if you see, among others:
```
CXI_FORK_SAFE_HP=1
FI_CXI_DISABLE_CQ_HUGETLB=1
FI_HMEM=1
FI_LOG_LEVEL=warn
FI_LOG_PROV=cxi
FI_PROVIDER=cxi
HSA_ENABLE_DEBUG=1
HSA_FORCE_FINE_GRAIN_PCIE=1
MIOPEN_DISABLE_CACHE=1
MIOPEN_USER_DB_PATH=
NCCL_NET_GDR_LEVEL=PHB
NCCL_SOCKET_IFNAME=hsn
PLUGIN_DIR=/project/project_foo/foo/rccl-lib
RCCL_DEBUG=INFO
RCCL_ENABLE_DMABUF_PLUGIN=0
RCCL_MSCCL_ENABLE=1
RCCL_NET_GDR_LEVEL=PHB
RCCL_TRACE_PLUGIN=1
```
This raises some useful observations (Thanks to ChatGPT 5.0):

- #1 fix: `FI_HMEM=1` is wrong. Libfabric expects interface names, e.g. rocr/cuda/ze, not a boolean. That’s why you get “unknown HMEM interface … entry='1'”. Set: `export FI_HMEM=rocr` (or just unset and let it auto-detect). 
- HSN selection: You currently use `NCCL_SOCKET_IFNAME=hsn`, and your logs shows IP mismatches on `hsn1-3`. Be explicit to avoid the warnings by setting: `export NCCL_SOCKET_IFNAME=hsn0`. LUMI guidance often shows listing even all (`hsn0,hsn1,hsn2,hsn3`) HSN ports, this has to be done correctly to work.
[see](https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf?utm_source=chatgpt.com)
- You’ve got RCCL_ENABLE_DMABUF_PLUGIN=0 (disabled). Given your earlier segfault with NCCL_DMABUF_ENABLE=1, I’d keep DMABUF off (don’t set NCCL_DMABUF_ENABLE) until you align versions (aws-ofi-rccl vs RCCL vs libfabric). Recent notes highlight API/version mismatches; the plugin commonly targets ncclNet v5.
olcf.ornl.gov
- CXI_FORK_SAFE_HP=1 and FI_CXI_DISABLE_CQ_HUGETLB=1 are fine (and commonly recommended on Slingshot to avoid fork/hugepage issues with PyTorch dataloaders/containers). Keep them. 
- Debug agents: You have HSA_ENABLE_DEBUG=1. That enables the ROC debug agent hooks and **can add overhead**; unset it unless you’re actively debugging:
`unset HSA_ENABLE_DEBUG`. (ROCm Documentation)
- Fine-grain PCIe: `HSA_FORCE_FINE_GRAIN_PCIE=1` is only needed if your code/tests allocate fine-grained PCIe memory (some rccl-tests do). Otherwise you can leave it, or unset if you want the default behavior. (ROCm Documentation)
- MIOpen caching: You’ve disabled it (`MIOPEN_DISABLE_CACHE=1`) and left `MIOPEN_USER_DB_PATH` empty. That slows conv algo selection. Prefer enabling cache and pointing it to a job-local dir (Any writable tmp dir is fine.):
```
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_USER_DB_PATH=/scratch/$PROJECT/$USER/miopen-cache-${SLURM_JOB_ID}
```
- GDR level: `NCCL_NET_GDR_LEVEL=PHB` / `RCCL_NET_GDR_LEVEL=PHB` are sane defaults and match AMD networking guidance (instinct.docs.amd.com)
- Noise control: `RCCL_DEBUG=INFO` + `RCCL_TRACE_PLUGIN=1` are chatty; drop to WARN (or VERSION for a quick version print) when you’re done debugging.

Suggested settings:
```
# Libfabric + Slingshot
export FI_PROVIDER=cxi
export FI_HMEM=rocr
export FI_LOG_LEVEL=warn
export FI_LOG_PROV=cxi
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

# NCCL/RCCL networking
export NCCL_SOCKET_IFNAME=hsn0            # or: hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export RCCL_NET_GDR_LEVEL=PHB
unset NCCL_DMABUF_ENABLE                   # keep DMABUF off for now

# ROCm
unset HSA_ENABLE_DEBUG
# only if you need fine-grained PCIe memory:
# export HSA_FORCE_FINE_GRAIN_PCIE=1

# MIOpen cache (optional but recommended for perf)
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_USER_DB_PATH=/tmp/$USER-miopen-${SLURM_JOB_ID}
```
