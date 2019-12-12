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

![Image of Yaktocat](https://raw.githubusercontent.com/Devenpor572/CleanEnergyHVACGroup/master/resources/temperature_reward.png)

This function was designed to model comfortable temperatures.

The action cost is simply -1 if action is taken, 0 if no action is taken. This is to simulate the cost of turning on an HVAC unit in real life.

## Learner

The learner is found in `hvac_learner/hvac_learner.py`. The code for the learner largely came from this article:

[Cartpole - Introduction to Reinforcement Learning](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288)

[Cartpole Learner Code](https://github.com/gsurma/cartpole)

The learner logs its state to a CSV file every step. This in turn may be rendered to a plot.

## Tester

I created a test script `gym_hvac_tester/tester.py` to help fine tune the parameters of the model. It non-intelligently steps the model forward while following a sequence of actions specified using command line parameters. The action sequence is encoded using run-length encoding.

## Plotters

The plotters were created to create plots for the final presentation. They use pandas, seaborn, and Matplotlib to to plot temperature trends, as well as other trends.

## References

- [Making a custom environment in gym](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)
- [Cartpole - Introduction to Reinforcement Learning](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288)
- [Heating/Cooling Differential Equations](http://www.sharetechnote.com/html/DE_Modeling_Example_Cooling.html)
