# RCCL-tests

This repository is a small ad hoc collection of small tools for testing RCCL functionalities on an AMD cluster, such as lumi.csc.fi.
The repository runs in user space and it is solely based on python and bash scripts and the availability of some python libraries, such as pytorch.

## The usage

Log in to lumi and set your node, environment, python, pytorch, virtual environment and environment variables as you wish.  Change to the directory of you clone of this repo and run:  `python rccl_test.py`.

This test was developed for the MARMoT project that will need multiprocessor communication.  In this project, one could write:
```
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
module use $PROJHOME/modules      # this directory contains modules of my project
module load pytorch-rocm-mammoth  # you can load also some pytorch module
python rccl_test.py
```
If your are not a part of the project, you can run load pytorch as follows:
```
module load LUMI
module load partition/C
module load cray-python
module use /appl/local/csc/modulefiles/
module load pytorch
```

Whatever you do, you need a version of Python and PyTorch installed.  Then, you need to be on a computing node.

```
srun --account=project_462000964 --partition=dev-g --ntasks=1 --gres=gpu:mi250:1 --time=00:10:00 --pty bash
echo $SLURM_JOB_PARTITION
```

## Output

Depending on the context, you may get different kinds of outputs that can help you to diagnose how your RCCL connections work.  Here are some example usages.

### 1. Running on a login node

```
rccl_test: 🚀 RCCL + OFI Communication Test (LUMI version)
rccl_test: ------------------------------------------------------------
rccl_test:     ❌ No ROCm GPU detected by PyTorch.
```

### 2. Running on computing node without extra settings

```
rccl_test: 🚀 RCCL + OFI Communication Test (LUMI version)
rccl_test: ------------------------------------------------------------
rccl_test:     ✅ ROCm GPU successfully detected by PyTorch.
rccl_test: 🔍 Container binding check:
rccl_test:     📦 Detected a _singularity_ container environment.
rccl_test:     ✅ ROCm bindings (/dev/{kfd,dri} and /opt/rocm) appear to be available.
rccl_test:     🔢 ROCm detected: AMD clang version 18.0.0git (...)
rccl_test: 🔍 Verifying loaded RCCL plugin via environment:
rccl_test:     PLUGIN_DIR = /project/project_462000964/members/aylijyra/rccl-lib3.10
rccl_test:     LD_LIBRARY_PATH = /usr/local/lib:/opt/rocm/lib/:...
rccl_test:     ❌ RCCL OFI plugin not found in LD_LIBRARY_PATH.
rccl_test: 🔍 Kernel parameter check:
rccl_test:     ⚠️ Kernel boot params might be missing or alternate: amd_iommu=on
rccl_test:        although related params are present: iommu=pt
rccl_test: 🔍 DMA Buffering Capacity Check:
rccl_test:     ✅ ROCm version rocm-core6.2.0-6.2.0.60200-sles155.66.x86_64
rocm-core-6.2.0.60200-sles155.66.x86_64 > 6.0, as required by DMABUF
rccl_test:     ✅ System may be DMA-BUF capable
rccl_test:     ❌ We are in a container
rccl_test: 🔍 FI_HMEM Diagnostic ===
rccl_test:    FI_HMEM = 1
rccl_test:    ✅ FI_HMEM=1 is set — Libfabric HMEM support should be enabled.
rccl_test:    ✅ Torch ROCm tensor ops working: y[0] = 2.0
/bin/sh: fi_info: command not found
rccl_test:    ❌ fi_info check failed: Command 'fi_info -p cxi -v' returned non-zero exit status 127.
rccl_test: 🔍 NCCL_SOCKET_IFNAME Diagnostic
rccl_test:    ✅ NCCL_SOCKET_IFNAME = hsn
/bin/sh: ip: command not found
rccl_test:    ❌ No interfaces found starting with 'hsn'
rccl_test: ✅ Successfully initialized RCCL backend.
```

This test report indicates some success.  In addition
- It gives a warning about kernel parameters.  This is not a problem.
- It observes that DMA buffering is not turned on.  Earlier, running a container was a problem with it.  Nevertheless, we still lack evidence that DMA buffering has an effect on performance.
- There are errors relato the availability pf `fi_info` and `ip` command. This affects the test, but can be fixed by bindings that are currently lacking from the given pytorch module.
- The test script reports an error with `NCCL_SOCKET_IFNAME`.  This should work on LUMI, but there seems to be a problem with the script or something.

In addition, the output contains the diagnostic log:
```
nid007971:55487:55487 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to hsn
nid007971:55487:55487 [0] NCCL INFO Bootstrap : Using hsn0:10.253.1.200<0>
nid007971:55487:55487 [0] NCCL INFO Plugin name set by env to librccl-net-ofi.so
nid007971:55487:55487 [0] NCCL INFO NET/Plugin : dlerror=librccl-net-ofi.so: cannot open shared object file: No such file or directory No plugin found (librccl-net-ofi.so), using internal implementation
nid007971:55487:55487 [0] NCCL INFO Kernel version: 5.14.21-150500.55.49_13.0.56-cray_shasta_c
nid007971:55487:55487 [0] NCCL INFO RCCL_MSCCL_ENABLE set by environment to 1.
nid007971:55487:55487 [0] NCCL INFO ROCr version 1.14
nid007971:55487:55487 [0] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
RCCL version 2.20.5+hip6.2 HEAD:45b618a+
nid007971:55487:55971 [0] NCCL INFO Failed to open libibverbs.so[.1]
nid007971:55487:55971 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to hsn
nid007971:55487:55971 [0] NCCL INFO NET/Socket : Using [0]hsn0:10.253.1.200<0> [1]hsn1:10.253.1.199<0> [2]hsn2:10.253.1.183<0> [3]hsn3:10.253.1.184<0>
nid007971:55487:55971 [0] NCCL INFO Using non-device net plugin version 0
nid007971:55487:55971 [0] NCCL INFO Using network Socket
nid007971:55487:55971 [0] NCCL INFO comm 0x8535a70 rank 0 nranks 1 cudaDev 0 busId d6000 commId 0x863e165bb2de6a89 - Init START
...
```
The log indicates that
- `NCCL_SOCKET_IFNAME` is set correctly although the test did not detect this.  NET/Socket is using hsn0...hsn3
- The plugin file `librccl-net-ofi.so` was not found.  The system fail back to an internal implementation.  This may be sufficient, if the implementation enables OFI, but unfortunate, if this fails back to an older and slower protocoll.  We do not know from the log what happens.  Some LUMI super-users say that the plugin is already handled by them.  This cannot be observed from the log.
- `RCCL_MSCCL_ENABLE` is set
- ROCr version is 1.14
- Dmabuf feature is disabled

### Running on a compute node with extra settings




