#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import time
import numpy as np
import argparse

argparser = argparse.ArgumentParser(
        description=__doc__)
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default='5555',
    help='TCP port of CARLA Simulator (default: 5555)')
args = argparser.parse_args()

context = zmq.Context()


#  Socket to talk to server
print("Connecting to worker server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + args.port)
# socket.connect("tcp://140.113.240.44:5555")

#  Do 10 requests, waiting each time for a response
# for request in range(10):
# socket.send_string(("%d,%d" % (7, 0)))
# exit()
time_list = list()
TTC_list = list()
for i in range(0, 10):
    TTC_fixed_wp = list()
    elapse_time_list = list()
    for j in range (-60, 70, 10):
        start_time = time.time()
        request = '1'
        param1 = i
        param2 = j
        print("Sending request %s …" % request)
        print(f'param {param1}, {param2}')
        # send a task to a worker
        socket.send_string(("%d,%d" % (param1, param2)))

        #  Get the reply.
        message = socket.recv()
        elapse_time = time.time() - start_time
        print("Received reply %s [ %s ]" % (request, message))
        encoding = 'utf-8'
        message = message.decode(encoding)
        TTC_str = message.split(',')[1]
        TTC_fixed_wp.append((float(TTC_str)))
        
        print('elapse_time:', elapse_time)
        
        elapse_time_list.append(elapse_time)
    time_list.append(elapse_time_list)
    TTC_list.append(TTC_fixed_wp)
    # np.save('TTC_list_wp=%d.npy' % (i), TTC_fixed_wp) 
    # np.save('elapse_time_list_wp=%d.npy' % (i), elapse_time_list) 

# np.save('time_list.npy', time_list)
# np.save('TTC_list.npy', time_list) #(10,13)
print(time_list)
print(np.mean(time_list))