### Testing the Torch version and CUDA availability

The test script is called `torch_env_check2.py`.  Here is a usage example:

```bash
srun --account=project_462000964 --partition=dev-g --ntasks=1 --gres=gpu:mi250:1 --time=00:30:00 --pty bash
echo $SLURM_JOB_PARTITION
source init3.10-with-conda.sh
# module load LUMI/23.09  partition/L
# module use /appl/local/containers/ai-modules
# module load singularity-AI-bindings                # AI bindings will be needed
# export SIF=$PROJHOME/sif/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif
singularity exec $SIF bash
$WITH_CONDA
python  $PROJHOME/diag/torch_env_check2.py
```

This outputs the following:
```bash
=== Torch ROCm/CUDA Diagnostic Script ===
Detected platform: LUMI
=== PyTorch Environment Check ===
PyTorch version          : 2.3.0+rocm6.2.0
ROCm HIP version          : 6.2.41133-dd7f95766
torch.version.cuda        : None
torch.cuda.is_available() : True
torch.cuda.device_count() : 1
Device 0: AMD Instinct MI250X
=== Validation ===
✅ LUMI: ROCm detected, no CUDA version reported — environment is OK.
```

I get the same result if I set the environment variables as recommended and redo the test.
Thus, this test is not very effective in normal usage, but it can give some information and do some basic overall checks.
