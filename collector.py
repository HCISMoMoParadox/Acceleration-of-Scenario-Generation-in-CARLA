# Task sink
# Binds PULL socket to tcp://localhost:5558
# Collects results from workers via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>

import sys
import time
import zmq
import os
import json
from datetime import datetime as dt

sample_size = 20

context = zmq.Context()

# Socket to receive messages on
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5558")

# receiver = context.socket(zmq.PULL)
# receiver.bind("tcp://*:5559")

# Wait for start of batch
# s = receiver.recv()

# Start our clock now
# tstart = time.time()

# Process 100 confirmations
counter = 0
TTC_list = list()
config_list = list()

dt_object = dt.fromtimestamp(dt.timestamp(dt.now()))
cfg_file_name = f"config_{dt_object.day}_{dt_object.hour:02d}-{dt_object.minute:02d}-{dt_object.second:02d}.json"


if not os.path.exists('work_result.json'):
    with open('work_result.json', 'w') as file:
        json.dump({}, file, indent=4, sort_keys=True)
        file.close()

work_dict = dict()


while True:
    message = receiver.recv()
    encoding = 'utf-8'
    message = message.decode(encoding)

    # print("Received reply [ %s ]" % (message))
    params = message.split(',')

    config_str = params[3:]
    configs = {}
    for config in config_str:
        key, value = config.split(':')
        configs[key] = value

    config_list.append(configs)

    print(params)
    # collided,TTC,work_num
    work_num = int(params[2])
    collided = True if params[0] == 'True' else False
    TTC = float(params[1])
    work_dict[work_num] = {'collided': collided, 'TTC': TTC}

    file = open('./cem/work_result.json', 'w')
    json.dump(work_dict, file, indent=4, sort_keys=True)
    file.close()

    with open(cfg_file_name, 'w') as f:
        json.dump(config_list, f, indent=4)

    counter += 1
    if counter == sample_size:
        assert(len(work_dict.keys()) == sample_size)
        work_dict = dict()
        counter = 0
        time.sleep(10)
        file.close()
        while True:
            with open('./cem/work_result.json', 'r') as f:
                work_result = json.loads(f.read())  # read
                f.close()
                if len(work_result.keys()) == 0:
                    break
                else:
                    time.sleep(3)
