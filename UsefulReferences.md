
# Useful References

## IBM Reinforcement Learning Testbed for EnergyPlus - [Paper](https://arxiv.org/abs/1808.10427); [Code](https://github.com/IBM/rl-testbed-for-energyplus)

_A testbed<sup>*</sup> for Energy Plus_

**DISCLAIMER**: I wouldn't get our hopes up until we can actually run their code on our computers.

### Introduction

IBM researchers created an open source reinforcement learning "testbed" for EnergyPlus to help them mitigate HVAC energy costs for datacenters. However, it appears that the only datacenter specific information used by them may be the input IDF file, which means this framework might be perfect for our use. This framework is designed to be used in conjuction with OpenGym. They've designed it to use the following API:
 - Env.reset(): Restart the simulation process
 - Env.step(): Proceed one simulation timestep
 
 The paper describes exactly how we can implement reinforcement learning using their framework.
 
 ### Risks
 
  - It seems pretty complicated, so it may be more involved than just plug and play, but we'll see.
  - Their framework has only been tested on the following OSs:
    - macOS High Sierra (Version 10.13.6)
    - Ubuntu 16.04.2 LTS, 18.04.2 LTS

\* _Testbed_ (noun.) A piece of equipment used for testing new machinery, especially aircraft engines.


## Input output reference

Here is the link to [PDF](https://www.energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v9.2.0/InputOutputReference.pdf)

We will mostly be focusing on chapter 2 of the document (pg. 2355). We will be working with the HVAC templates. Chapter 2 goes through the values in the templates, for applicable values, the default is documented here


## Eppy simple tutorial
This [tutorial](https://eppy.readthedocs.io/en/latest/Main_Tutorial.html#modifying-idf-fields)  goes over the basics of using eppy to edit the .idf files. This will be a key part for our simulation. The agent will modify the state through the idf file with eppy.

This [link](https://eppy.readthedocs.io/en/latest/runningeplus.html) talks about how to run an eppy simulation from a python source file


