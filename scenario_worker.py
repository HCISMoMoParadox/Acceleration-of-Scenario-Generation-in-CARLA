import glob
import os
import sys
import random
import time
import csv
import json
from tracemalloc import start
from turtle import back

# try:
    #
sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
#
# sys.path.append('../carla/agents/navigation')
sys.path.append('../carla/agents')
sys.path.append('../carla/')
    # sys.path.append('../../HDMaps')
    # sys.path.append('rss/') # rss

# except IndexError:
#     pass

import carla
import pygame
import cv2
import weakref


from scenario import *
from read_input import *
from get_and_control_trafficlight import *
from LBC.lbc_utils import CollisionSensor
from controller import VehiclePIDController
from carla import ColorConverter as cc

DEBUG = 2 #1: DataAgent 2: LBCAgent
RECORD = 0
SHOW_PYGAME = 0

##############################################
##Uitility Functions##########################
##############################################
def set_bp(blueprint, actor_id):
    blueprint = random.choice(blueprint)
    blueprint.set_attribute('role_name', 'tp')
    if blueprint.has_attribute('color'):
        color = random.choice(
            blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    if blueprint.has_attribute('driver_id'):
        driver_id = random.choice(
            blueprint.get_attribute('driver_id').recommended_values)
        blueprint.set_attribute('driver_id', driver_id)
    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'true')

    return blueprint

def render(surface, display):
    if surface is not None:
        display.blit(surface, (0, 0))
##############################################
##############################################
##############################################

class CarlaEnv(object):
    def __init__(self, start_index, degree_deff, port=2000):
        scenario_root = './SCSG_test/5_i-1_0_c_l_f_1_0'
        # scenario_root = './SCSG_test/3_r-1_0_c_sr_r_1_0_1'
        # scenario_root = '/home/momoparadox/carla13_RSS/CARLA_0.9.13_RSS/PythonAPI/SCSG/SCSG_test/3_i-5_1_c_l_l_1_0'
        # scenario_root = '/home/momoparadox/carla13_RSS/CARLA_0.9.13_RSS/PythonAPI/SCSG/SCSG_test/5_i-7_1_c_l_f_1_0'
        # scenario_root = '/home/momoparadox/carla13_RSS/CARLA_0.9.13_RSS/PythonAPI/SCSG/SCSG_test/3_r-1_0_c_sr_r_1_0_1'
        # scenario_root = '/home/momoparadox/carla13_RSS/CARLA_0.9.13_RSS/PythonAPI/SCSG/SCSG_test/3_t1-2_0_m_f_l_1_0'


        self.scenario_ = Scenario(scenario_root)
        
        # adjust start idnex
        if start_index > 0:
            for i in range(start_index):
                self.scenario_.ego_car_.transform_.pop(0)
        # adjust orientation
        self.scenario_.ego_car_.transform_[0].rotation.yaw += degree_deff
        self.scenario_.ego_car_.init_transform_ = self.scenario_.ego_car_.transform_[0]



        if RECORD:
            self.out = cv2.VideoWriter('scenario_init_video'+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,  (1280, 720) )
        # self.filter_dict = {}
        # try:
        #     for root, _, files in os.walk(scenario_root + '/filter/'):
        #         for name in files:
        #             f = open(scenario_root + '/filter/' + name, 'r')
        #             bp = f.readlines()[0]
        #             name = name.strip('.txt')
        #             f.close()
        #             self.filter_dict[name] = bp
        #     print(self.filter_dict)
        # except:
        #     print("no scenario file")

        # pygame initialization
        pygame.init()
        pygame.font.init()
        if SHOW_PYGAME:
            display_width = 1280
            display_height = 720
            self.display = pygame.display.set_mode(
            (display_width, display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0, 0, 0))
            pygame.display.flip()
        self.clock = pygame.time.Clock()

        # # load files for scenario reproducing
        # self.transform_dict = {}
        # self.velocity_dict = {}
        # self.ped_control_dict = {}
        # for actor_id, filter in self.filter_dict.items():
        #     self.transform_dict[actor_id] = read_transform(
        #         os.path.join(scenario_root, 'transform', actor_id + '.npy'))
        #     self.velocity_dict[actor_id] = read_velocity(
        #         os.path.join(scenario_root, 'velocity', actor_id + '.npy'))
        #     if 'pedestrian' in filter:
        #         self.ped_control_dict[actor_id] = read_ped_control(
        #             os.path.join(scenario_root, 'ped_control', actor_id + '.npy'))
        

        # self.agents_dict = {}
        # self.controller_dict = {}
        # self.actor_transform_index = {}
        # self.finish = {}
        

        # connect to a carla host, initialization for the whole scenarios
        try:
            self.client = carla.Client('127.0.0.1', port)
            self.client.set_timeout(10.0)
            self.sim_world = self.client.get_world()

            # sync mode
            settings = self.sim_world.get_settings()
            settings.fixed_delta_seconds = 0.05  #0.05
            settings.synchronous_mode = True  # Enables synchronous mode
            self.sim_world.apply_settings(settings)

            # blueprint_library = self.client.get_world().get_blueprint_library()

            # read traffic light
            self.lights = []
            actors = self.sim_world.get_actors().filter('traffic.traffic_light*')
            for l in actors:
                self.lights.append(l)
            self.light_dict, self.light_transform_dict = read_traffic_lights(scenario_root, self.lights)

            # # spawn ego car at init. position
            # model_3 = blueprint_library.filter("model3")[0]
            # random_transform = random.choice(self.sim_world.get_map().get_spawn_points())
            # self.ego_car = self.sim_world.spawn_actor(model_3, random_transform)
            # # self.actor_list.append(self.ego_car)

            # ego_transform = self.transform_dict['player'][0]
            # ego_transform.location.z += 3

            # self.ego_car.set_transform(ego_transform)
            # self.agents_dict['player'] = self.ego_car

            # # set controller
            # for actor_id, bp in self.filter_dict.items():
            #     if actor_id != 'player':
            #         transform_spawn = self.transform_dict[actor_id][0]
                    
            #         while True:
            #             try:
            #                 self.agents_dict[actor_id] = self.sim_world.spawn_actor(
            #                     set_bp(blueprint_library.filter(
            #                         self.filter_dict[actor_id]), actor_id),
            #                     transform_spawn)

            #                 break
            #             except Exception:
            #                 transform_spawn.location.z += 1.5

            #         # set other actor id for checking collision object's identity
            #         # world.collision_sensor.other_actor_id = agents_dict[actor_id].id

            #     if 'vehicle' in bp:
            #         self.controller_dict[actor_id] = VehiclePIDController(self.agents_dict[actor_id], args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
            #                                                             max_throttle=1.0, max_brake=1.0, max_steering=1.0)
            #         try:
            #             self.agents_dict[actor_id].set_light_state(carla.VehicleLightState.LowBeam)
            #         except:
            #             print('vehicle has no low beam light')
            #     self.actor_transform_index[actor_id] = 1
            #     self.finish[actor_id] = False

            # init. ego car and tps
            if DEBUG == 1:
                self.ego_agent = DataAgent(self.scenario_.ego_car_, self.sim_world)
            if DEBUG == 2:
                self.ego_agent = LBCAgent(self.scenario_.ego_car_, self.sim_world)
            # print('here')

            self.tp_agent_list = []
            for id, tp_data in self.scenario_.tps_.items():
                # print(id, tp_data)
                tp_agent = DataAgent(tp_data, self.sim_world)
                self.tp_agent_list.append(tp_agent)
            
            # print(self.tp_agent_list)

            # sensor setup
            bound_x = 0.5 + self.ego_agent.actor_.bounding_box.extent.x
            bound_y = 0.5 + self.ego_agent.actor_.bounding_box.extent.y
            bound_z = 0.5 + self.ego_agent.actor_.bounding_box.extent.z
            Attachment = carla.AttachmentType
            # top view

            topview_transform = carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                z=23*bound_z), carla.Rotation(pitch=18.0))

            cemara_blueprint = self.sim_world.get_blueprint_library().find('sensor.camera.rgb')
            cemara_blueprint.set_attribute('image_size_x', str(1280))
            cemara_blueprint.set_attribute('image_size_y', str(720))

            if SHOW_PYGAME or RECORD:
                self.sensor_top = self.sim_world.spawn_actor(
                    cemara_blueprint,
                    topview_transform,
                    attach_to=self.ego_agent.actor_,
                    attachment_type=Attachment.SpringArm)
                weak_self = weakref.ref(self)
                self.sensor_top.listen(
                    lambda image: CarlaEnv._parse_image(weak_self, image, 'top'))
                self.surface = None

            # collision sensor
            self.collision_sensor = CollisionSensor(self.ego_agent.actor_)

            # tick to update world
            self.frame = self.sim_world.tick()
            print('done carla init.')
        finally:
            pass

    def step(self):
        # print(self.ego_agent.actor_)
        # print(self.light_transform_dict)
        # print(self.sim_world)
        # ref_light = get_next_traffic_light(
        #     self.ego_agent.actor_, self.sim_world, self.light_transform_dict)
        # # print(ref_light)
        # if ref_light:
        #     annotate = annotate_trafficlight_in_group(
        #         ref_light, self.lights, self.sim_world)

        # set_light_state(self.lights, self.light_dict,
        #             self.ego_agent.index_, annotate)
        # for actor_id, _ in self.filter_dict.items():
        #     # apply recorded location and velocity on the controller
        #     actors = self.sim_world.get_actors()
        #     # reproduce traffic light state
        #     if actor_id == 'player' and ref_light:
        #         set_light_state(
        #             self.lights, self.light_dict, self.actor_transform_index[actor_id], annotate)

        #     if self.actor_transform_index[actor_id] < len(self.transform_dict[actor_id]):
        #         # x = self.transform_dict[actor_id][self.actor_transform_index[actor_id]].location.x
        #         # y = self.transform_dict[actor_id][self.actor_transform_index[actor_id]].location.y
        #         if 'vehicle' in self.filter_dict[actor_id]:

        #             target_speed = (self.velocity_dict[actor_id][self.actor_transform_index[actor_id]])*3.6
        #             waypoint = self.transform_dict[actor_id][self.actor_transform_index[actor_id]]

        #             self.agents_dict[actor_id].apply_control(self.controller_dict[actor_id].run_step(target_speed, waypoint))                            
        #             # agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(
        #             #     (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6, transform_dict[actor_id][actor_transform_index[actor_id]]))

                    # v = self.agents_dict[actor_id].get_velocity()
                    # v = ((v.x)**2 + (v.y)**2+(v.z)**2)**(0.5)

                    # # to avoid the actor slowing down for the dense location around
                    # # if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2 + v/20.0:
                    # if self.agents_dict[actor_id].get_transform().location.distance(self.transform_dict[actor_id][self.actor_transform_index[actor_id]].location) < 2.0:
                    #     self.actor_transform_index[actor_id] += 2
                    # elif self.agents_dict[actor_id].get_transform().location.distance(self.transform_dict[actor_id][self.actor_transform_index[actor_id]].location) > 6.0:
                    #     self.actor_transform_index[actor_id] += 6
                    # else:
                    #     self.actor_transform_index[actor_id] += 1

        #         elif 'pedestrian' in self.filter_dict[actor_id]:
        #                     self.agents_dict[actor_id].apply_control(
        #                         self.ped_control_dict[actor_id][self.actor_transform_index[actor_id]])
        #                     self.actor_transform_index[actor_id] += 1

        #     else:
        #         self.finish[actor_id] = True

        # choose action ###
        if DEBUG == 1:
            self.ego_agent.choose_action() # DataAgent
        if DEBUG == 2:
            self.ego_agent.choose_action(self.frame) # LBCAgent

        for tp_agent in self.tp_agent_list:
            tp_agent.choose_action()
        ###################

        # render ##########
        if SHOW_PYGAME:
            self.clock.tick_busy_loop(20)            
        self.frame = self.sim_world.tick()   # run the action
        if SHOW_PYGAME:
            render(self.surface, self.display)
            pygame.display.flip()
        if RECORD and self.surface:
            pygame.image.save(self.surface, "screenshot.jpeg")
            image = cv2.imread("screenshot.jpeg")
            self.out.write(image)
        ###################
        
        # if False in self.finish.values():
        #     return False
        # else:
        #     return True
        if not RECORD and len(self.collision_sensor.history) > 0:
            return True

        if not self.ego_agent.finish_:
            return False
        else:
            return True
        


    def end(self):
        # resume to async mode
        settings = self.sim_world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = False  # Enables synchronous mode
        self.sim_world.apply_settings(settings)

        if RECORD:
            self.out.release()
        pygame.quit()

        
    
    @staticmethod
    def _parse_image(weak_self, image, view='top'):
        self = weak_self()
        if view == 'top':
            image.convert(cc.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # render the view shown in monitor
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def run_scenario_with_params(params, port):
    env_init = 0
    env_step = 0
    env_TTC = 0
    env_end = 0
    time_last = time.time()
    env = CarlaEnv(params[0], params[1], port)
    time_end = time.time()
    env_init = time_end - time_last
    # print('env init: %d second' % (env_init))


    TTC_list = []
    
    start_time = time.time()
    while True:
        
        if time.time() - start_time > 30:
            break
        # env.choose_action()
        time_last = time.time()
        done = env.step()
        time_end = time.time()
        env_this_step = time_end - time_last
        env_step += env_this_step
        # print('env step: %d second' % (env_this_step))

        # collect TTC
        time_last = time.time()
        for agent in env.tp_agent_list:
            ego_loc = env.ego_agent.actor_.get_location()
            agent_loc = agent.actor_.get_location()
            rel_loc = carla.Vector3D(agent_loc.x - ego_loc.x, agent_loc.y - ego_loc.y, agent_loc.z - ego_loc.z)

            ego_vel = env.ego_agent.actor_.get_velocity()
            agent_vel = agent.actor_.get_velocity()
            rel_vel = carla.Vector3D(agent_vel.x - ego_vel.x, agent_vel.y - ego_vel.y, agent_vel.z - ego_vel.z)            
            if rel_vel.length() != 0 and rel_loc.dot(rel_vel) < 0:
                TTC_list.append(abs(rel_loc.length() ** 2 / rel_loc.dot(rel_vel)))
            # if rel_vel.length() != 0:
            #     abs_ttc = rel_loc.length() / rel_vel.length()
            #     if rel_loc.dot(rel_vel) >= 0:
            #         TTC_list.append(abs_ttc)
            #     else:
            #         TTC_list.append(-abs_ttc)
        time_end = time.time()
        env_this_TTC = time_end - time_last
        env_TTC += env_this_TTC
        # print('safety criterion: %d second' % (env_this_TTC))
        #
        if done:
            # save result of this scenario
            break
    # end of a scenario
    # RES iteration reward update

    # destroy all agents in carla world
    collided = "False"
    if len(env.collision_sensor.history) > 0:
        collided = "True"
    time_last = time.time()
    if SHOW_PYGAME or RECORD:
        env.sensor_top.destroy()
    env.collision_sensor.sensor.destroy()
    env.ego_agent.destroy()
    for agent in env.tp_agent_list:
        agent.destroy()

    env.end()
    time_end = time.time()
    env_end = time_end - time_last
    # print('env end: %d second' % (env_end))


    print('===========================')
    print('env init: %d second' % (env_init))
    print('env step: %d second' % (env_step))
    print('safety criterion: %d second' % (env_TTC))
    print('env end: %d second' % (env_end))
    print('===========================')

    total_sec = env_init + env_step + env_TTC + env_end
    port_list.append(total_sec)

    # print(TTC_list)
    min_TTC = min( [i for i in TTC_list if i > 0])
    # print('min_TTC', min_TTC)
    # import psutil
    # proc = psutil.Process()
    # print (proc.open_files())
    return "%s,%f" % (collided, min_TTC)

if __name__ == '__main__':
    # params = [0, 0]
    # run_scenario_with_params(params)
    
    import zmq
    import argparse

    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:5557")

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5558")

    # main_argparser = argparse.ArgumentParser(
    #     description=__doc__)
    # main_argparser.add_argument(
    # '-p', '--port',
    # metavar='P',
    # default=2000,
    # type=int,
    # help='TCP port of CARLA Simulator (default: 2000)')
    # if len(sys.argv) < 2:
    #     main_argparser.print_help()
    #     exit()
    
    # args = main_argparser.parse_args()
    print('argv[2]:', sys.argv[2])

    port1 = []
    port2 = []
    port3 = []
    
    while True:
        #  Wait for next request from client
        message = receiver.recv()
        print("Received request: %s" % message)
        print('argv[2]:', sys.argv[2])
        encoding = 'utf-8'
        message = message.decode(encoding)
        params = message.split(',')
        print(params)
        
        # params = [int(a) for a in params]
        params[0] = int(params[0])
        params[1] = float(params[1])
        params[2] = int(params[2])

        # print(sys.argv[1])
        
        # param1, param2, work_num

        #  Do some 'work'
        #params = message
        result = "87"
        if len(params) == 3:
            # if sys.argv[2]==2000:
            #     result= run_scenario_with_params(params[0:2], int(sys.argv[2]), port1)
            # elif sys.argv[2]==2002:
            #     result= run_scenario_with_params(params[0:2], int(sys.argv[2]), port2)
            # elif sys.argv[2]==2004:
            #     result= run_scenario_with_params(params[0:2], int(sys.argv[2]), port3)
            # else:
            #     print("Port Error.")
            run_scenario_with_params(params[0:2], int(sys.argv[2]))
            result = result + ',' +str(params[2])
            # collided,TTC,work_num

        #  Send reply back to client
        sender.send(result.encode())

    # # RES iteration loop
    # # run a scenario
    # env = CarlaEnv()

    # TTC_list = []

    # while True:
    #     # env.choose_action()
    #     done = env.step()

    #     # collect TTC
    #     for agent in env.tp_agent_list:
    #         ego_loc = env.ego_agent.actor_.get_location()
    #         agent_loc = agent.actor_.get_location()
    #         rel_loc = carla.Vector3D(agent_loc.x - ego_loc.x, agent_loc.y - ego_loc.y, agent_loc.z - ego_loc.z)

    #         ego_vel = env.ego_agent.actor_.get_velocity()
    #         agent_vel = agent.actor_.get_velocity()
    #         rel_vel = carla.Vector3D(agent_vel.x - ego_vel.x, agent_vel.y - ego_vel.y, agent_vel.z - ego_vel.z)            

    #         if rel_vel.length() != 0:
    #             abs_ttc = rel_loc.length() / rel_vel.length()
    #             if rel_loc.dot(rel_vel) >= 0:
    #                 TTC_list.append(abs_ttc)
    #             else:
    #                 TTC_list.append(-abs_ttc)

    #     #
    #     if done:
    #         # save result of this scenario
    #         break
    # # end of a scenario
    # # RES iteration reward update

    # # destroy all agents in carla world
    # env.sensor_top.destroy()
    # env.ego_agent.destroy()
    # for agent in env.tp_agent_list:
    #     agent.destroy()
    
    # env.end()
    
    # print(TTC_list)
    # print('min_TTC', min( [i for i in TTC_list if i > 0]))
    

    # RES iteration loop
    # run a scenario
    import numpy as np
    import time

    # for start_index in range(9, 10, 1):
    #     for degree_diff in range (-30, 70, 10):
    env = CarlaEnv(0, 0)


    TTC_list = []
    start_time = time.time()

    while True:
        if time.time() - start_time > 30:
            break
        # env.choose_action()
        done = env.step()

        # collect TTC
        for agent in env.tp_agent_list:
            ego_loc = env.ego_agent.actor_.get_location()
            agent_loc = agent.actor_.get_location()
            rel_loc = carla.Vector3D(agent_loc.x - ego_loc.x, agent_loc.y - ego_loc.y, agent_loc.z - ego_loc.z)

            ego_vel = env.ego_agent.actor_.get_velocity()
            agent_vel = agent.actor_.get_velocity()
            rel_vel = carla.Vector3D(agent_vel.x - ego_vel.x, agent_vel.y - ego_vel.y, agent_vel.z - ego_vel.z)            
            if rel_vel.length() != 0 and rel_loc.dot(rel_vel) < 0:
                TTC_list.append(abs(rel_loc.length() ** 2 / rel_loc.dot(rel_vel)))
            # if rel_vel.length() != 0:
            #     abs_ttc = rel_loc.length() / rel_vel.length()
            #     if rel_loc.dot(rel_vel) >= 0:
            #         TTC_list.append(abs_ttc)
            #     else:
            #         TTC_list.append(-abs_ttc)

        #
        if done:
            # save result of this scenario
            break
    # end of a scenario
    # RES iteration reward update

    # destroy all agents in carla world
    env.sensor_top.destroy()
    env.collision_sensor.sensor.destroy()
    env.ego_agent.destroy()
    for agent in env.tp_agent_list:
        agent.destroy()

    env.end()

    # print(TTC_list)
    min_TTC = min( [i for i in TTC_list if i > 0])
    print('min_TTC', min_TTC)
    exit()
            # try:
            #     a = np.load('experiment_result.npy',allow_pickle=True).tolist()
            # except OSError: # no file exist
            #     a = []
            # finally:
            #     a.append([start_index, degree_diff, min_TTC, env.collision_sensor.history])
            #     np.save('experiment_result.npy', a)



    scenario_root = 232
    criterion = 1321


    # run a scenario
    scenario = Scenario(scenario_root)
    env = CarlaEnv(scenario)

    while True:
        done = env.step()
        criterion()
        if done:
            break
    
    # print("Port1 total time= ", np.sum(port1))
    # print("Port2 total time= ", np.sum(port2))
    # print("Port3 total time= ", np.sum(port3))
    
    # np.save('port1.npy', np.array(port1))
    # np.save('port2.npy', np.array(port2))
    # np.save('port3.npy', np.array(port3))
