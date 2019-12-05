import configparser

cfg = configparser.ConfigParser()
cfg.read('resources/env_config.ini')

for section in cfg.sections():
    print('Skip Section {}? y/N'.format(section))
    skip = ''
    while skip != 'y' and skip != 'N':
        skip = input()
    if skip == 'y':
        continue
    for field in cfg[section]:
        print('\t{} = {}'.format(field, cfg[section][field]))
        overwrite = input('\t')
        if overwrite:
            cfg[section][field] = overwrite
            print('[U]\t{} = {}'.format(field, cfg[section][field]))

with open('resources/env_config.ini', 'w') as configfile:
    cfg.write(configfile)
