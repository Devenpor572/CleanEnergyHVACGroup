"""
HVAC system following the classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from collections import namedtuple


class HVACEnv(gym.Env):
    """
    Description:
        A home with three rooms: A basement, an attic, and the surrounding air.
        The basement is the lowest room. It touches the earth and the main floor.
        The main floor is the middle room. It touches the basement, the attic, and the surrounding air.
        The attic is upper room. It touches the main floor and the surrounding air.

    Source:
        http://www.sharetechnote.com/html/DE_Modeling_Example_Cooling.html

    Observation:
        Type: Box(5)
        Num	Observation                 Min         Max
        0	Temperature Air             -273        Inf
        1	Temperature Ground          -273        Inf
        2	Temperature HVAC            -273        Inf
        3	Temperature Basement        0           40
        4	Temperature Main Floor      0           40
        5	Temperature Attic           0           40

    "30 is hot, 20 is pleasing, 10 is cold, 0 is freezing"
    20 Celsius (68 F) is roughly room temperature, and 30 and 10 make convenient hot/cold thresholds.

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Turn the cooler on
        1   Turn everything off
        2	Turn the heater on

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [10..20]

    Episode Termination:
        Temperature Basement is less than 10 or more than 30
        Temperature Main Floor is less than 10 or more than 30
        Temperature Attic is less than 10 or more than 30
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    # k - The cooling constant of the boundary
    # t - Function to get the temperature of the other side of the boundary
    # w - The weight of the boundary. Can be used to give relative size of boundary
    Boundary = namedtuple('Boundary', ['k', 't', 'w'])

    class Room(object):
        def __init__(self, boundary_list=None, hvac=None):
            self.boundary_list = boundary_list
            self.hvac = hvac

        def get_temp_change_eq(self):
            def temp_change_eq(time, current_temp, action):
                return sum([boundary.k * boundary.w * (current_temp - boundary.t(time))
                            for boundary in self.boundary_list]) \
                       + (self.hvac(action) if self.hvac is not None else 0)
            return temp_change_eq

    @staticmethod
    # TODO FIND AN ACCEPTABLE VALUE FOR THIS CONSTANT
    def get_hvac(action):
        return (action - 1) * 5

    @staticmethod
    def temperature_ground(time):
        # Very rough estimate, but the ground temperature appears to be about 10 on average
        return 10

    @staticmethod
    def temperature_air(time):
        # This could be where weather data could come in.
        # For now just use 0 (or 40)
        return 0

    def __init__(self):
        def get_temperature_basement(time):
            return self.state[0]

        def get_temperature_main(time):
            return self.state[1]

        def get_temperature_attic(time):
            return self.state[2]

        self.basement = HVACEnv.Room(boundary_list=[
            # Basement-Earth Boundary
            # k is roughly = 0.25/hr,
            # The weight is a cube where 5 of the 6 sides are below ground)
            HVACEnv.Boundary(0.0000694, HVACEnv.temperature_ground, (5/6)),
            # Basement-Main Boundary
            # k is roughly = 0.5/hr,
            # The weight is a cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(0.0001389, get_temperature_main, (1/6))
        ])

        self.main = HVACEnv.Room(boundary_list=[
            # Main-Basement Boundary
            # k is roughly = 0.5/hr,
            # The weight is a cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(0.0001389, get_temperature_basement, (1 / 6)),
            # Basement-Earth Boundary
            # k is roughly = 0.25/hr,
            # The weight is a cube where 4 of the 6 sides are below ground)
            HVACEnv.Boundary(0.0000694, HVACEnv.temperature_air, (4 / 6)),
            # Main-Attic Boundary
            # k is roughly = 0.5/hr,
            # The weight is a cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(0.0001389, get_temperature_attic, (1 / 6))
        ], )

        self.attic = HVACEnv.Room(boundary_list=[
            # Main-Attic Boundary
            # k is roughly = 0.5/hr,
            # The weight is a cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(0.0001389, get_temperature_main, (1 / 6)),
            # Basement-Earth Boundary
            # k is roughly = 0.25/hr,
            # The weight is a cube where 5 of the 6 sides are below ground)
            HVACEnv.Boundary(0.0000694, HVACEnv.temperature_air, (5 / 6))
        ])

        # Thresholds at which to fail the episode
        self.desired_tempertaure_low = 20
        self.desired_tempertaure_high = 23
        self.lower_temperature_threshold = 10
        self.upper_temperature_threshold = 33

        '''
        Action space
            Num	Action
            0	Turn the cooler on
            1   No action
            2	Turn the heater on
        '''
        self.action_space = spaces.Discrete(3)

        '''
        Observation Space
            Num	Observation                 Min         Max
            0	Temperature Air             -273        Inf
            1	Temperature Ground          -273        Inf
            2	Temperature HVAC            -273        Inf
            3	Temperature Basement        0           40
            4	Temperature Main Floor      0           40
            5	Temperature Attic           0           40
        '''
        low = np.array([
            -273,
            -273,
            -273,
            0,
            0,
            0])

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            40,
            40,
            40])

        self.time = 0

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Calculate reward using this continuous function
    # y = -0.8165 * sqrt(abs(x - 21.5)) + 1
    # This function was chosen/created because at room temperature (21.5 celsius) it gives a reward of +1,
    # at the thresholds of comfort (roughly 20 to 23 celsius) it returns 0,
    # and around the minimum and maximum threshold (10 and 33 celsius) it returns roughly -1.75, which isn't too extreme
    # In the range 20-23 just use reward 1.
    @staticmethod
    def calculate_temperature_reward(state):
        reward = 0
        for temperature in state.tolist()[3:]:
            if 20 <= temperature <= 23:
                reward += 1
            else:
                reward += -0.8165 * math.sqrt(abs(temperature - 21.5)) + 1
        return reward

    @staticmethod
    def calculate_action_cost(action):
        return -1 if action != 1 else 0

    # The weights 0.75 and 0.25 are arbitrary, but we probably don't want the learner to gain too much from no action
    @staticmethod
    def calculate_reward(state, action):
        return 0.75 * HVACEnv.calculate_temperature_reward(state) + 0.25 * HVACEnv.calculate_action_cost(action)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        air_temp, ground_temp, hvac_temp, basement_temp, main_temp, attic_temp = state

        # Basement
        basement_temp_change_equation = self.basement.get_temp_change_eq()
        new_basement_temp = basement_temp_change_equation(self.time, basement_temp, action) + basement_temp

        # Main
        main_temp_change_equation = self.main.get_temp_change_eq()
        new_main_temp = main_temp_change_equation(self.time, main_temp, action) + main_temp

        # Attic
        attic_temp_change_equation = self.attic.get_temp_change_eq()
        new_attic_temp = attic_temp_change_equation(self.time, attic_temp, action) + attic_temp

        # Calculate done
        self.state = (self.temperature_air(self.time),
                      self.temperature_ground(self.time),
                      HVACEnv.get_hvac(action),
                      new_basement_temp,
                      new_main_temp,
                      new_attic_temp)

        done = self.lower_temperature_threshold > new_basement_temp > self.upper_temperature_threshold \
            or self.lower_temperature_threshold > new_main_temp > self.upper_temperature_threshold \
            or self.lower_temperature_threshold > new_attic_temp > self.upper_temperature_threshold
        done = bool(done)

        if not done:
            reward = HVACEnv.calculate_reward(state, action)
        elif self.steps_beyond_done is None:
            # Episode just ended!
            self.steps_beyond_done = 0
            reward = HVACEnv.calculate_reward(state, action)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        self.time += 1

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.concatenate((np.array([self.temperature_air(0),
                                               self.temperature_ground(0),
                                               0]),
                                     self.np_random.uniform(low=10, high=30, size=(3,))), axis=0)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
