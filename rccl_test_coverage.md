## Test Coverage Matrix

* ✅ Tested **by `rccl_test.py`** (directly in Python),
* 📄 Validated **via log messages**, or
* ❓ Not easily testable / indirect.

This split table shows what aspects of the multiprocessing environment are currently tested by `rccl_test.py`.  I have added explanations and quotations to emphasize the importance of some of the features and variables since they can increase the interprocess communication between the nodes and betweem GPUs, or slow down that with extra log messages.  There is more to do in the latter: the table now shows mainly how to get more log messages to debug, but you may want to turn these messages off to avoid the overhead.  

The table shows that many things are not being checked by the current tool.  This is mainly OK, but inconvenient.  Furthermore, some features can only be observed as settings, but not through the logs.  They may require performance probing -- not the purpose of the current tool, unless there is a great need for that.

### Overall functionality
| Variable / Setting            | Purpose / Effect                                                       | Tested by `rccl_test2.py` | Validated via Logs | Notes / Observability                              |
| ----------------------------- | ---------------------------------------------------------------------- | :-----------------------: | :----------------: | -------------------------------------------------- |
| ROCm availability             | Whether GPU + `/dev/kfd`, `/dev/dri`, `/opt/rocm` are usable           |             ✅             |     📄 Partial     | Script verifies bindings inside container          |
| ROCm version                  | Verifies ROCm version (must be ≥ 6.0 for some features)                |             ✅             |       📄 Yes       | Used in DMA-BUF test                               |
| Container awareness           | Whether running inside a Singularity container                         |             ✅             |     📄 Partial     | Impacts DMA-BUF and `/dev` access                  |
| RCCL plugin file present      | Checks `librccl-net-ofi.so` exists and can be loaded                   |             ✅             |       📄 Yes       | Symbol presence checked in logs                    |
| All-reduce result             | Validates basic collective communication (RCCL works)                  |             ✅             |       📄 Yes       | Correct (42.0) result of the rccl_test.pu confirms it                            |

### RCCL AWS-CXI plugin: Libfabric
"Comms are important! RCCL AWS-CXI plugin enables collectives computation on devices (3-4x faster collectives). Minimizes the role of the CPU in the control path – expose more asynchronous computation opportunities. Lowest latency for network message passing is from GPU HBM memory. Requires: **HPE Cray Libfabric** implementation from (https://github.com/ROCm/aws-ofi-rccl).  Included in the LUMI provided containers! If not using the LUMI containers make sure you have that in your environment with these settings."
See (https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf)

Related settings:  `CXI_FORK_SAFE_HP=1` and `FI_CXI_DISABLE_CQ_HUGETLB=1` are fine (and commonly recommended on Slingshot to avoid fork/hugepage issues with PyTorch dataloaders/containers). Keep them.  (docs.nersc.gov, lumi-supercomputer.github.io)

| Variable / Setting            | Purpose / Effect                                                       | Tested by `rccl_test2.py` | Validated via Logs | Notes / Observability                              |
| ----------------------------- | ---------------------------------------------------------------------- | :-----------------------: | :----------------: | -------------------------------------------------- |
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT` | Enables verbose RCCL logging. Search the logs for: `NCCL INFO NET/OFI Using aws-ofi-rccl 1.4.0`    |             ❌             |       📄 Yes       | Log level shown in RCCL/NCCL output. Is `NCCL_DEBUG_SUBSYS=INIT` available in standard LUMI RCCL builds? |
| `RCCL_TRACE_PLUGIN=1`         | Enables plugin tracing for RCCL                                        |             ❌             |     📄 Partial     | Log shows plugin symbol loading/failures           |
| `PLUGIN_DIR`                  | Specifies location for RCCL plugins                                    |             ✅             |       📄 Yes       | Python finds `.so`, confirms symlink               |
| `LD_LIBRARY_PATH`             | Required to find dynamic `.so` files like `libfabric.so` and ``.       |             ✅             |       📄 Yes       | Logged paths shown; plugin linked successfully     |
| `RCCL_ENABLE_OFI=1`           | Enables RCCL OFI (libfabric) plugin for inter-node GPU comm            |             ✅             |       📄 Yes       | Plugin presence + symbol load shown in logs        |
| `FI_PROVIDER=cxi`             | Selects libfabric CXI (Slingshot) provider                             |             ✅             |       📄 Yes       | Logs show provider, or error if not found          |
| `FI_PROVIDER_PATH`            | Path to locate FI providers (like cxi) if not in default               |             ❌             |      📄 Maybe      | Not always set explicitly — fallback path shown    |
| `FI_LOG_LEVEL=debug`          | Debug-level logs from libfabric                                        |             ❌             |       📄 Yes       | In log, helps trace `FI_HMEM` and other diagnostics |
| `CXI_FORK_SAFE_HP=1` | Slingshot to avoid fork/hugepage issues with PyTorch dataloaders/containers | | | Commonly recommended |
| `FI_CXI_DISABLE_CQ_HUGETLB=1´ | Slingshot to avoid fork/hugepage issues with PyTorch dataloaders/containers | | | Commonly recommended |
| `FI_HMEM=rocr`                   | Enables Libfabric's HMEM for GPU memory communication.                   |             ✅             |       📄 Yes       | Also tested via `fi_info` if available. Note that `FI_HMEM=1` is not correct for LUMI. The value must be `rocr/cuda/ze`.             |

#### Summary of the Perceived Recommendations: 
```
LD_LIBRARY_PATH=<set path>   # To find dynamic `.so` files like libfabric.so
RCCL_ENABLE_OFI=1            # Enables RCCL OFI (libfabric) plugin for inter-node GPU comm 
FI_PROVIDER=cxi              # Selects libfabric CXI (Slingshot) provider
CXI_FORK_SAFE_HP=1           # Slingshot, to avoid fork/hugepage issues with PyTorch
FI_CXI_DISABLE_CQ_HUGETLB=1  # Slingshot, to avoid fork/hugepage issues with PyTorch
FI_HMEM=rocr                 # Enables Libfabric's HMEM for GPU memory communication
```

### Node-Local Caches
"Just-in-time compiles are a common technique in AI applications. MIOpen leverages this functionality.  MIOpen is a library for high-optimized machine learning primitives. Used on many models – not in our LLM example though. It uses caches to enable just-in-time compilation organized as SQLite databases.  File system doesn’t deal well with SQLite locks when many processes are trying to access it.
Solution? Setup individual caches for groups of ranks – we recommend per node.  Let’s cache those builds in node-local storage instead of the default home folder." 
See (https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf)
| Variable / Setting            | Purpose / Effect                                                       | Tested by `rccl_test2.py` | Validated via Logs | Notes / Observability                              |
| ----------------------------- | ---------------------------------------------------------------------- | :-----------------------: | :----------------: | -------------------------------------------------- |
| `MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-\$SLURM_NODEID"` | A node-local storage |  |  |  |
| `MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH`                    | Use to cache just-in-time MPI compiles |  |  |  |
| `MIOPEN_ENABLE_LOGGING=0`     | Turn this 1 to chack MPI activity  |  |  |  |                    
| `RCCL_MSCCL_ENABLE=1`         | Enables MSCCL plugin (multi-source collectives)                        |             ❌             |       📄 Yes       | RCCL logs confirm MSCCL activation                 |

#### Summary of the Perceived Recommendations: 
```
MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-\$SLURM_NODEID"
MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH  # Use a node-local storage to cache just-in-time MPI compiles
MIOPEN_ENABLE_LOGGING=0                        # Turn this 1 to chack MPI activity
RCCL_MSCCL_ENABLE=1                            # Enables MSCCL plugin (multi-source collectives)
```

### Point RCCL to use the high-speed network interfaces
"RCCL should be set to use only high-speed-interfaces - Slingshot.  Point RCCL to use all 4 high-speed interfaces by setting `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3` or (`NCCL_SOCKET_IFNAME=hsn`). It will know how to bind them based on the node topology."

"RCCL should be set configured to use GPU RDMA by setting `NCCL_NET_GDR_LEVEL=PHB`.  On ROCm versions (6.2) this is not needed – it is
default. Careful using external containers as you may need to be setting plugin yourself!  2x better bandwidth utilization with RDMA = can scale further!"  This feature is dependent on kernel parameters that can be checked for completeness of the analysis but that are not for the user to modify.

See (https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf)
| Variable / Setting            | Purpose / Effect                                                       | Tested by `rccl_test2.py` | Validated via Logs | Notes / Observability                              |
| ----------------------------- | ---------------------------------------------------------------------- | :-----------------------: | :----------------: | -------------------------------------------------- |
| `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hns3` | Tells NCCL/RCCL to use Slingshot HSN network interface                 |             ✅             |       📄 Yes       | Interface existence + IP match checked             |
| `NCCL_DEBUG=INFO` | Enables verbose RCCL logging. Shows if node has interfaces other than Slingshot    |             ❌             |       📄 Yes       | Log level shown in RCCL/NCCL output
| `NCCL_NET_GDR_LEVEL=PHB`      | Enables GPU Direct RDMA when GPU and NIC share PCIe Host Bridge        |             ❌             |        ❓ No        | Not visible in logs, no standard method to check. On ROCm versions (6.2) this is not needed – it is default.   |
| Kernel Param: `amd_iommu=on`  | Enables AMD IOMMU (required for RDMA and GPU memory mapping)           |         ✅ Inferred        |     📄 Partial     | `iommu=pt` seen instead; acceptable on LUMI        |
| Kernel Param: `iommu=pt`      | Pass-through IOMMU mode (enables devices but no memory remapping)      |         ✅ Inferred        |       📄 Yes       | Warning issued if `amd_iommu=on` missing           |

#### Summary of the Perceived Recommendations: 
```
NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hns3 # Tells NCCL/RCCL to use Slingshot High Speed Network interfaces
NCCL_NET_GDR_LEVEL=PHB                 # Enables GPU Direct RDMA when GPU and NIC share PCIe Host Bridge
```


### Not Usable: DMA Buffering Plugin
Given your earlier segfault with NCCL_DMABUF_ENABLE=1, I’d keep DMABUF off (don’t set NCCL_DMABUF_ENABLE) until you align versions (aws-ofi-rccl vs RCCL vs libfabric). Recent notes highlight API/version mismatches; the plugin commonly targets ncclNet v5. (ChatGPT5, olcf.ornl.gov)
| Variable / Setting            | Purpose / Effect                                                       | Tested by `rccl_test2.py` | Validated via Logs | Notes / Observability                              |
| ----------------------------- | ---------------------------------------------------------------------- | :-----------------------: | :----------------: | -------------------------------------------------- |
| `RCCL_ENABLE_DMABUF_PLUGIN=0` | Disables DMA-BUF GPU interop (set to 0 since containers lack bindings) |             ✅             |       📄 Yes       | Log confirms feature is disabled                   |

### Fine-grain PCIe
`HSA_FORCE_FINE_GRAIN_PCIE=1` is only needed if your code/tests allocate fine-grained PCIe memory (some rccl-tests do). Otherwise you can leave it, or unset if you want the default behavior.   
| Variable / Setting            | Purpose / Effect                                                       | Tested by `rccl_test2.py` | Validated via Logs | Notes / Observability                              |
| ----------------------------- | ---------------------------------------------------------------------- | :-----------------------: | :----------------: | -------------------------------------------------- |
| `HSA_FORCE_FINE_GRAIN_PCIE=0` | Enables fine-grained PCIe memory — ROCm zero-copy memory behavior      |         ✅ Partial         |          ❌         | Indirectly inferred from tensor success            |

### Debug agents: 
If you have `HSA_ENABLE_DEBUG=1`, that enables the ROC debug agent hooks and can add overhead; unset it unless you’re actively debugging:
`unset HSA_ENABLE_DEBUG`. [ROCm Documentation: ROCR Debug Agent user guide](https://rocm.docs.amd.com/projects/rocr_debug_agent/en/docs-6.2.2/conceptual/user-guide.html)

| Variable / Setting            | Purpose / Effect                                                       | Tested by `rccl_test2.py` | Validated via Logs | Notes / Observability                              |
| ----------------------------- | ---------------------------------------------------------------------- | :-----------------------: | :----------------: | -------------------------------------------------- |
| `HSA_ENABLE_DEBUG=0`          | Enables HSA debug hooks (no visible effect unless debug image used)    |             ❌             |          ❌         | Usually ignored in production containers           |
| `HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2` | Debug tracing lib for HSA — not usually shipped in containers          |             ❌             |     📄 Warning     | Expected to fail if not present (harmless)         |


* ✅ = tested by Python script logic (`rccl_test2.py`)
* 📄 = confirmed via log files (`stdout`, `stderr`, `RCCL INFO`, etc.)
* ❌ = not tested or intentionally skipped
* ❓ = indirect or difficult to verify (no logging or public interface)
