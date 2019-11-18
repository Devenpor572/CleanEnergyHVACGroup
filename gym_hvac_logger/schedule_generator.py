import json

count = 100

data = dict()
data['lower_temperature_threshold'] = -1000
data['upper_temperature_threshold'] = 1000
data['ground_temperature'] = 10
data['air_temperature'] = 0
data['hvac_temperature'] = 0.00333
data['basement_temperature'] = 15
data['main_temperature'] = 20
data['attic_temperature'] = 25
data['tau'] = 300

data['action_schedule'] = [2 for i in range(count)]

with open('output/schedule.json', 'w') as outfile:
    json.dump(data, outfile)
