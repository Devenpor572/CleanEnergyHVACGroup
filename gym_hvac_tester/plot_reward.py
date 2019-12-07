import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np

# Create the vectors X and Y


desired_temperature_low = 20
desired_temperature_high = 23


def reward_func(temperature):
    reward = 0
    if desired_temperature_low <= temperature <= desired_temperature_high:
        reward += 1
    elif temperature < desired_temperature_low:
        reward += -0.8165 * math.sqrt(abs(temperature - desired_temperature_low)) + 1
    # Temperature > desired temperature high
    else:
        reward += -0.8165 * math.sqrt(abs(temperature - desired_temperature_high)) + 1
    return reward


# Create temperature-reward plot
sub_increments = 4
x = np.array([el / sub_increments for el in range(0, 43 * sub_increments)])
y = np.array([reward_func(el) for el in x])
fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Reward')
ax.set_title('Temperature Reward Function')
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.plot(x, y)

# Add 23 celsius marker
xt = ax.get_xticks()
xt = np.append(xt, 23)
xtl = xt.tolist()
for idx, tick in enumerate(xtl):
    xtl[idx] = str(int(tick))
xtl[-1] = '23'
ax.set_xticks(xt)
ax.set_xticklabels(xtl)

# Create room temperature area
ax.axvspan(desired_temperature_low, desired_temperature_high, color='#36D7B7', alpha=0.25)

# Create y=0
ax.plot(x, np.array([0 for el in x]), color='black')

# Show the plot
plt.tight_layout()
plt.savefig('output/temperature_reward.png')
plt.show()
plt.close()

