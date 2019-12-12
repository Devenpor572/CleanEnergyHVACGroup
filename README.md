# HVAC Optimization Using Reinforcement Learning

## Introduction

_DISCLAIMER: This README is directed at future Computer Science students at Utah State University that may want to continue this project._

Smart HVAC systems are increasing in importance as the climate changes. In the next 30 years, the demand for air conditioning is expected to increase three-fold as the Earth warms and warm countries develop. This project uses reinforcement learning to optimize a smart thermostat. 


## Setup

This project was run using `Python 3.6.8` for compatibility with Keras, Tensorflow, etc...

I froze all the packages I was using with their versions in requirements.txt. These can be installed with `pip install -r requirements.txt`.

To install the custom environment, go to the `gym-hvac` folder and type `pip install -e .` 

The main script is `hvac_learner.py` which runs the learning loop indefinitely and logs its results to `hvac_learner/output/results.csv`.

## Environment

### Overview

This project uses a custom OpenGym environment to simulate heating and cooling in a home. The custom gym environment is located in `gym-hvac/gym_hvac/envs/hvac_env.py/`

### Basic Elements

#### Observation Space

These are observations the learner receives from the environment. In a realistic scenario, only the room/rooms with thermostats would be observable.

| Num | Observation            | Min  | Max |
|-----|------------------------|------|-----|
| 0   | Temperature Air        | -273 | Inf |
| 1   | Temperature Ground     | -273 | Inf |
| 2   | Temperature HVAC       | -273 | Inf |
| 3   | Temperature Basement   | 0    | 40  |
| 4   | Temperature Main Floor | 0    | 40  |
| 5   | Temperature Attic      | 0    | 40  |

#### Action Space

These are the possible actions the learner can take to modify the environment. In a realistic scenario it wouldn't make sense to turn on the heater when it's warmer than room temperature outside and vice versa.

| Num | Action                              |
|-----|-------------------------------------|
| 0   | Turn the cooler on                  |
| 1   | No action or turn off heater/cooler |
| 2   | Turn the heater on                  |

#### Initialization

The initialization function sets up the static environment. This includes the rooms which are modeled as a set of boundaries (walls) and optionally an HVAC unit. Currently the main floor is the only floor with an HVAC unit. The init function also sets up the weather generator object (which isn't actually a Python generator). The weather generator keeps track of weather patterns found in `gym-hvac/gym_hvac/envs/resources/weather.csv`. This weather data is hourly weather data near Utah State University from 2013-2019.

#### Reset Function

The reset function resets the state of the environment. It also generates a new state from the weather data.

#### Step Function

The step function updates internal temperatures, determines if termination conditions have been met (too hot, too cold, or too many steps), and calculates the reward. 

#### Reward Function

The reward function consists of two parts: the temperature reward and the action cost. These two parts are summed together using a weighted function.
 
The temperature reward function is modeled using this image:

This function was designed to model comfortable temperatures.

The action cost is simply -1 if action is taken, 0 if no action is taken. This is to simulate the cost of turning on an HVAC unit in real life.



### References

I used this article as a reference while generating the custom environment.

[Making a custom environment in gym](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)




[Cartpole - Introduction to Reinforcement Learning](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288)
  



## Usage Guide

Resources I used for the custom OpenGym environment
 


## Learner

