# RCCL-tests

This repository is a small ad hoc collection of small tools for testing RCCL functionalities on an AMD cluster, such as lumi.csc.fi.

## DISCLAIMER

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

- `rccl_test.py` - the most comprehensive test
- `test_fine_grain2.py` - testing fine grained memory (not very useful), see `test_fine_grain.md`
- `torch_env_check2.py` - testing torch (a simpler variant of `rccl_test.py`)
- `envcheck.sh` - a stupid way to check comms relevant environment variables

## Licence
Most of the tests are simple enough to be written in 5 minutes or a day.  I have used ChatGPT to help me writing some of the test code.  Thus, the tentative licence for this software is Creative Commons.  You are free to use the code as you wish, but I would be vary happy to get improved versions of the tools back.

