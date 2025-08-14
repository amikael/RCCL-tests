# RCCL-tests

This repository is a small ad hoc collection of small tools for testing RCCL functionalities on an AMD cluster, such as lumi.csc.fi.
A better name for this software would have been `comms_tests` since it is all about comms, i.e., interprocess communications.  However, that name would have been a bit too promising since the amount of tests based on life communications is currently minimal.  The main focus of the current tests is just to see that the intended and recommended environment variable settings are set when the commands are running and that they cause the intended effect on initialization of the interprocess communications.

## DISCLAIMER

This software is not meant to be shared or published as a part of my work.  I made this originally for myself and I do not expect that this software is useful for anybody else.  However, at some point I thought that it may be useful to keep track of these explorations.  Putting the software to GitHub is one of best ways to store and develop the software for an extended period of time without forgetting it to somewhere in my directories.

The software has been authored using ChatGPT to help with details.  You may use this software only at your risk.  The repository is for running tests as an ordinary user.  The risks of damage should be limited to the user's files and processes.  It is solely based on python and bash scripts and the availability of some python libraries, such as pytorch.  The tests are rather straightforward and easy to review for possible vulnerabilities.  

## The Philosophy behind the Tools

1. There are standard best practices to use LUMI correctly, with CSC tested containers and modules.  These come with promises of supported ROCm and NCCL/RCCL features such as Slingshot, Librfabric etc.  
2. The user may fail to follow the best practices: although module system helps, there are ways to get in a trouble if one uses mismatching modules that are not supporting the latest ROCm etc.  Furthermore, modules, containers, sbatch files, and python programs can set environment variables before ROCm / Libfabric initialization.  In short, everything conveivable can still go wrong.
3. The current tests have been developed to help the users to gain better understanding of what is going on.  They try to show the current settings and/or test the ROCm initialisation given the current environment.
4. However, even tests need testing.  There is a possibility of false positives and false negatives.  For example, rccl_test.pu itself sets some environment variables.  It happened that some of these changes shadowed the current environment.   Furthermore, since pytorch module in LUMI has been implemented as a container that starts when a python command is given, you need to understand that the environment changes slightly when the python starts.  Note also that when the pytorch module is loaded, the environment changes drastically:  there are changes that anticipate the python command and the singularity startup, and the module sets MOST of the ROCm/NCCL/Libfabric related settings at this point to enable a decent environment of pytorch.

## The Use Cases 
Thus, there was a use case for the tool `rccl_test.py` and many simpler tools included into this repository.  For example, these were used
1. to check that the CSC provided modules are used correctly and the result is expectable
2. to check the effect of some environment settings (`rocm-setup.sh` to be sourced in sbatch scripts) done by the user in the interest of following CSC provided instructions and training and guidelines concerning Libfabric, Cache and Hight Speed Network (hsn) settings.
3. to check that a CSC provided singularity container is used correctly and the environment variables are set correctly by the user
4. to check that combining the above CSC resources into a new user-defined module (`pytorch-rocm-mammoth`) is doing the desirable difference in the environment

Since the role of the modules in this process is so high and since I ended up putting my settings to the user-defined module, the final use case is the most important for me.  However, if you not doing any user-defined module (it is tricky and unlikely), but you are concerned about the environment you may still benefit from using the test `rccl_test.sh`.

If you are a novice user and you just want to understand the difference between C/L nodes, the simpler tests (execpt `test_fine_grain2.sh`) can be useful for learning about the environment.

## What Tools Are There?
The directory contains the following tools and the respective documentation:

- `rccl_test.py` - the most comprehensive test, see [`rccl_test_coverage.md`](rccl_test_coverage.md) for its coverage
- `test_fine_grain2.py` - testing fine grained memory (not very useful), see `test_fine_grain.md`
- `torch_env_check2.py` - testing torch (a simpler variant of `rccl_test.py`)
- `envcheck.sh` - a stupid way to check comms relevant environment variables

## What tools are still missing?

- A tool that would automatically analyse your logfiles and report what works and what needs fixing.  Would be extremely useful, since currently I have had to learn various codes to look for in the log file.  Alternatively, when I have not understood something, I have asked ChatGPT.  That is a lot of waste of time and CO2.  I encourage to develop such a bash-sourceable script (just a set of ifs) yourself and send it to me as a contribution.
  
- A tool that does the same for the environment.  It would be very easy to develop such a shell snippet that could be sourced in a batchfile before launching the `srun`.  It would simply read and test the values of a bunch of environment variables and then do one of the following: (1) stop with a warning if your environment is underperforming, (2) do nothing if everything looks good, or (3) report what is fine and what is not fine (if you have a debug variable set).  I encourage to develop such a bash-sourceable script (just a set of ifs) yourself and send it to me as a contribution.

In the end, I think these two quick-to-develop tools would be the most useful ones since they could potentially be useful both for ordinary users and for superusers.  I just do not have enough time to develop them now.

## Licence
Most of the tests are simple enough to be written in 5 minutes or a day.  I have used ChatGPT to help me writing some of the test code.  Thus, the tentative licence for this software is Creative Commons.  You are free to use the code as you wish, but I would be vary happy to get improved versions of the tools back.

## Information about standard test methods that may be more appropriate for you
I try to list here some useful commands that are not part of this software but that you can use to do similar things.  For an experienced user, some of these commands can be much more useful, but for random user, it may take time to gather the information from the small pieces.  Some of these tests may also be handy in sh scripts (in fact used also in `rccl_test.py` and even in sbatch scripts.  

- To see wheter you are on a GPU node, you can query: `echo $SLURM_JOB_PARTITION`.  This is empty if you are on a login node.

- A "Smoke" test to confirm GPUs are available ([LUMI AI course material](https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf)):
  ```
  if [ \$SLURM_LOCALID -eq 0 ] ; then rocm-smi; fi
  ```
  Explanation:  It’s a Slurm-aware guard that runs `rocm-smi` once per node.  `SLURM_LOCALID` is the task’s local rank on the node (0,1,2,… up to --`ntasks-per-node`-1).  The test `if [ $SLURM_LOCALID -eq 0 ]` means: “only the task with local rank 0 on each node should run the command.”  Then `rocm-smi` prints GPU/ROCm status for the whole node, so running it once per node avoids N copies of the same output.

  Don’t escape the `$` in a normal script. Only escape if you’re generating the script from another shell and want to delay expansion.  Here is a slightly more robust test (Suggested by ChatGPT5):
  ```
  if [ "${SLURM_LOCALID:-0}" -eq 0 ]; then rocm-smi; fi
  ```
  
- Testing that you have RCCL AWS-CXI plugin correctly set up ([LUMI AI course material](https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf)):
  ```
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT
  ```
  Run your job and then search the logs for.
  ```
  [0] NCCL INFO NET/OFI Using aws-ofi-rccl 1.4.0
  ```
  This indicates that AWS RCCL Network plugin is in use.  

- Assuming that you have done `export NCCL_DEBUG=INFO`.  Then search from logs for `NCCL INFO NET/Socke` to see if High Speed Network is set up correctly.  You are using wrong protocols, if you see something like (Lumi AI course material):
  ```
  NCCL INFO NET/Socket : Using [0]nmn0:10.120.116.65<0> [1]hsn0:10.253.6.67<0>[2]hsn1:10.253.6.68<0> [3]hsn2:10.253.2.12<0> [4]hsn3:10.253.2.11<0>
  ```
  Basically, seeing `[1]hsn0:10.253.6.67<0>[2]hsn1:10.253.6.68<0> [3]hsn2:10.253.2.12<0> [4]hsn3:10.253.2.11<0>` is promising, but it is a bad sign to see slower networking interfaces being used: `0]nmn0:10.120.116.65<0>`.


There are many more manual tests, but I have not been able to list them here yet.
