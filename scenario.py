import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    
    sys.path.append('LBC/') # rss
except IndexError:
    pass
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import random
import carla

from read_input import *
from controller import VehiclePIDController
from get_and_control_trafficlight import *
from agent import Agent

# LBC
import torch
# from LBC.dataset import CarlaDataset
# from LBC.converter import Converter
from map_model import MapModel
from lbc_utils import *
import common as common
# import pathlib
# import uuid
import copy
# from collections import deque
import cv2
# import argparse
import weakref


##############################################
##Uitility Functions##########################
##############################################
def set_bp(blueprint):
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
##############################################
##############################################
##############################################


class DataAgent(Agent):
    def __init__(self, agent_data, world):
        '''
        Spawn agent in carla world.
        '''
        self.world_ = world
        self.data_ = agent_data

        blueprint_library = world.get_blueprint_library()
        if agent_data.type_ == 'ego':
            model_3 = blueprint_library.filter("model3")[0]
            random_transform = random.choice(world.get_map().get_spawn_points())
            self.actor_ = world.spawn_actor(model_3, random_transform)
            agent_data.init_transform_.location.z +=3
            self.actor_.set_transform(agent_data.init_transform_)

        else:
            while True:
                try:
                    self.actor_ = world.spawn_actor(
                                    set_bp(blueprint_library.filter(
                                        agent_data.blueprint_)), agent_data.init_transform_)
                    break
                except Exception:
                    agent_data.init_transform_.location.z += 1.5
        
        # print(agent_data.type_ ,'spawned')

        if 'vehicle' in agent_data.blueprint_:
            self.controller_ = VehiclePIDController(self.actor_, args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
                                                                        max_throttle=1.0, max_brake=1.0, max_steering=1.0)
            try:
                self.actor_.set_light_state(carla.VehicleLightState.LowBeam)
            except:
                print('vehicle has no low beam light')
        self.index_ = 2
        self.finish_ = False
    
    def choose_action(self):
        if not self.finish_:
            if 'vehicle' in self.data_.blueprint_:
                target_speed = (self.data_.velocity_[self.index_])*3.6
                target_transform = self.data_.transform_[self.index_]
                self.actor_.apply_control(self.controller_.run_step(target_speed, target_transform))

                distance_to_next_transform = self.actor_.get_transform().location.distance(self.data_.transform_[self.index_].location)
                if distance_to_next_transform < 2.0:
                    self.index_ += 2
                elif distance_to_next_transform > 6.0:
                    self.index_ += 6
                else:
                    self.index_ += 1

            elif 'pedestrian' in self.data_.blueprint_:
                self.actor_.apply_control(
                    self.data_.ped_control_dict[self.index_])
                self.index_ += 1

            if self.index_ >= len(self.data_.transform_):
                self.finish_ = True
    
    def destroy(self):
        self.actor_.destroy()

class LBCAgent(Agent):
    def __init__(self, agent_data, world):
        self.world_ = world
        self.data_ = agent_data

        blueprint_library = world.get_blueprint_library()
        if agent_data.type_ == 'ego':
            model_3 = blueprint_library.filter("model3")[0]
            random_transform = random.choice(world.get_map().get_spawn_points())
            self.actor_ = world.spawn_actor(model_3, random_transform)
            agent_data.init_transform_.location.z +=3
            self.actor_.set_transform(agent_data.init_transform_)

        else:
            while True:
                try:
                    self.actor_ = world.spawn_actor(set_bp(blueprint_library.filter(
                                        agent_data.blueprint_)), agent_data.init_transform_)
                    break
                except Exception:
                    agent_data.init_transform_.location.z += 1.5
        
        # print(agent_data.type_ ,'spawned')

        self.index_ = 1
        self.finish_ = False
        self.controller_ = LBCController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self.last_control_ = carla.VehicleControl()
        self.frame_counter_ = 0
        self.inference_fps = 4 # fps of world: 20 Hz
        self.target_ = agent_data.transform_[1]
        update_target(self)

        # load model
        parsed = LBCArgument()
        self.net_ = MapModel(parsed)
        self.net_ = MapModel.load_from_checkpoint('./LBC/epoch=34_01.ckpt')
        self.net_.cuda()
        self.net_.eval()

        # LBC sensor setup
        # bird-eye-view's semantic map
        self.bev_map_ = None
        self.bev_map_frame_ = None

        camera_transform = carla.Transform(carla.Location(x=0, y=0, z=100.0),
                            carla.Rotation(pitch=-90.0))
        bev_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        bev_seg_bp.set_attribute('image_size_x', str(512))
        bev_seg_bp.set_attribute('image_size_y', str(512))
        bev_seg_bp.set_attribute('fov', str(50.0))

        self.sensor_lbc_seg_ = world.spawn_actor(
                    bev_seg_bp,
                    camera_transform,
                    attach_to=self.actor_)

        weak_self = weakref.ref(self)
        self.sensor_lbc_seg_.listen(lambda image: LBCAgent._parse_image(weak_self, image))
        '''
        Read bird-eye-view semantic map from:
            self.bev_map_
            self.bev_map_frame_
        '''

        # IMU sensor
        self.imu_sensor_ = IMUSensor(self.actor_)
    
    def choose_action(self, frame):
        # frequency check
        if self.frame_counter_ % (20 // self.inference_fps):
            self.actor_.apply_control(self.last_control_)
            self.frame_counter_ += 1
        else:
            # update waypoint
            # min_distance = 7.5
            # max_distance = 25

            # waypoint_now = self.actor_.get_location()
            # if waypoint_now.transform.location.distance(self.target.transform.location) < min_distance:
            #     self.target = find_next_target(max_distance, waypoint_now, self)
                # sim_world.debug.draw_point(target.transform.location, 0.1, carla.Color(255,0,0), 0.0, True)
            update_target(self)
            target = self.target_
            
            # wait for bev map
            while True:    
                if self.bev_map_frame_ == frame:
                # if True:
                    topdown = self.bev_map_
                    # print('in_loop:', topdown)
                    break
            # wait for compass data(orientation)
            while True:    
                if self.imu_sensor_.frame == frame:
                # if True:
                    theta = self.imu_sensor_.compass
                    break
            # print(frame)
            # print(topdown.timestamp)
            theta = theta/180.0 * np.pi # degree to radius

            # topdown = self.bev_map_
            # theta = self.imu_sensor_.compass
            # theta = theta/180.0 * np.pi # degree to radius
            
            # print(theta)
            R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

            PIXELS_PER_WORLD = 5.5 # from coverter.py 
            ego_location = self.actor_.get_location()
            ego_xy = np.float32([ego_location.x, ego_location.y])
            # print(target)
            target_loc = target.location
            target_xy = np.float32([target_loc.x, target_loc.y])
            target_xy = R.T.dot(target_xy - ego_xy)
            target_xy_print = target_xy
            target_xy *= PIXELS_PER_WORLD
            # target_xy_print = target_xy
            target_xy += [128, 256]
            target_xy = np.clip(target_xy, 0, 256)
            target_xy = torch.FloatTensor(target_xy)

            # print('target', target_xy)
  
            
            # print('topdown.shape',topdown.shape)
            # topdown = topdown.crop((128, 0, 128 + 256, 256))
            topdown = topdown[0: 256, 128:384]
            # top_down_to_draw = copy.deepcopy(topdown)
            # print(topdown.shape)
            # print(topdown)
            
            topdown = np.array(topdown)
            topdown = common.CONVERTER[topdown]
            n_classes = len(common.COLOR)

            top_down_to_draw = copy.deepcopy(topdown) # pass to draw

            topdown = torch.LongTensor(topdown)
            topdown = torch.nn.functional.one_hot(topdown, n_classes).permute(2, 0, 1).float()
            # inference
            points_pred = self.net_.forward(topdown, target_xy)
            # print(points_pred)
            control = self.net_.controller.forward(points_pred)
            # print(control)
            

            # out, (target_heatmap,) = net.forward(topdown, target, debug=True)

            # alpha = torch.rand(out.shape).type_as(out)
            # between = alpha * out + (1-alpha) * points
            # out_cmd = net.controller.forward(between)
                    

            # alpha = torch.rand(out.shape).type_as(out)
            # between = alpha * out + (1-alpha) * points

            # draw points
            points_to_draw = points_pred.cpu().data.numpy()
            points_to_draw = points_to_draw * 256
            # print('points_to_draw:', points_to_draw)

            # points_to_draw -= [128, 256]
            # points_to_draw /= PIXELS_PER_WORLD
            # print(points_to_draw)
            R = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)],
            ])
            # print('R', R)
            # points_to_draw = (1/R.T).dot(points_to_draw[0], axis=1)
            points_draw = []
            points_draw.append(R.T.dot(points_to_draw[0][0]))
            points_draw.append(R.T.dot(points_to_draw[0][1]))
            points_draw.append(R.T.dot(points_to_draw[0][2]))
            points_draw.append(R.T.dot(points_to_draw[0][3]))
            # print('pred_points', points_draw)
            # sim_world.debug.draw_point(carla.Location(x=points_draw[0][0], y=points_draw[0][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
            # sim_world.debug.draw_point(carla.Location(x=points_draw[1][0], y=points_draw[1][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
            # sim_world.debug.draw_point(carla.Location(x=points_draw[2][0], y=points_draw[2][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
            # sim_world.debug.draw_point(carla.Location(x=points_draw[3][0], y=points_draw[3][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
            
            DRAW = False
            if DRAW == True:
                save_canvas = np.zeros((256, 256, 3), np.int8)
                # print('points_to_draw', points_to_draw)
                save_canvas = common.COLOR[top_down_to_draw]
                for xy in points_to_draw[0]:
                    # print(xy)
                    x_int = 128 + int(xy[0])
                    y_int = int(xy[1])
                    save_canvas[y_int][x_int] = [255, 255, 255]

                    x_target = int(target_xy_print[0]) if int(target_xy_print[0]) <= 255 else 255
                    y_target = int(target_xy_print[1]) if int(target_xy_print[1]) <= 255 else 255
                    save_canvas[y_target][x_target] = [0, 255, 0]
                    for pad in range(1, 3):
                        try:
                            save_canvas[y_int-pad][x_int+pad] = [255, 255, 255]
                            save_canvas[y_int+pad][x_int+pad] = [255, 255, 255]
                            save_canvas[y_int-pad][x_int-pad] = [255, 255, 255]
                            save_canvas[y_int+pad][x_int-pad] = [255, 255, 255]

                            save_canvas[y_target+pad][x_target+pad] = [0, 255, 0]
                            save_canvas[y_target+pad][x_target-pad] = [0, 255, 0]
                            save_canvas[y_target-pad][x_target+pad] = [0, 255, 0]
                            save_canvas[y_target-pad][x_target-pad] = [0, 255, 0]
                        except:
                            pass

                    # print(common.COLOR[top_down_to_draw[x_int][y_int]])
                cv2.imwrite('./test_result/'+str(frame)+'.png', cv2.cvtColor(save_canvas, cv2.COLOR_RGB2BGR))
                # out.write(cv2.cvtColor(save_canvas, cv2.COLOR_RGB2BGR))


            # control
            control = control.cpu().data.numpy()
            # print(control)
            steer = control[0][0] #* 1.4
            desired_speed = control[0][1]
            # print(steer, desired_speed)
            speed = self.actor_.get_velocity().length()

            brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1
            # brake = desired_speed < 0.4*1.4 or (speed / desired_speed) > 1.1 *1.4

            delta = np.clip(desired_speed - speed, 0.0, 0.25)
            throttle = self.controller_.step(delta)
            throttle = np.clip(throttle, 0.0, 0.75)
            throttle = throttle if not brake else 0.0

            self.last_control_.steer = float(steer)
            self.last_control_.throttle = float(throttle)
            self.last_control_.brake = float(brake)
            self.actor_.apply_control(self.last_control_)

            self.frame_counter_ += 1
        
        # terminal condition
        distance_to_end = self.actor_.get_transform().location.distance(self.data_.transform_[-1].location)
        if distance_to_end < 2:
            self.finish_ = True
        if distance_to_end > 50:
            self.finish_ = True

    def destroy(self):
        self.sensor_lbc_seg_.destroy()
        self.imu_sensor_.sensor.destroy()
        self.actor_.destroy()

    
    @staticmethod
    def _parse_image(weak_self, image, view='non-top'):
        '''
        Instance Variables: image
            fov (float - degrees): Horizontal field of view of the image.
            height (int): Image height in pixels.
            width (int): Image width in pixels.
            raw_data (bytes)
        '''
        self = weak_self()
        #take out the raw data
        array_raw = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        # print(array_raw)
        # print(array_raw.shape)
        array_raw = np.reshape(array_raw, (image.height, image.width, 4))
        actors = self.world_.get_actors()
        traffic_lights = get_nearby_lights(self.actor_, actors.filter('*traffic_light*'))
        array_raw = draw_traffic_lights(array_raw[:, :, 2], self.actor_, traffic_lights)
        self.bev_map_ = copy.deepcopy(array_raw)
        self.bev_map_frame_ = image.frame 

class AgentData(object):
    def __init__(self, type, blueprint):
        self.type_ = type
        self.blueprint_ = blueprint
    
    def set_transform(self, transform):
        self.transform_ = transform
        self.init_transform_ = transform[0]
    
    def set_velocity(self, velocity):
        self.velocity_ = velocity

    def set_ped_control(self, ped_control):
        self.ped_control_ = ped_control


class Scenario():
    def __init__(self, scenario_root):
        self.load(scenario_root)
    
    def load(self, scenario_root):
        '''
        Instantiate all the actors and the parameters of a scenario
        '''
        self.ego_car_ = None
        self.tps_ = {}
        # scenario_root = '/home/momoparadox/carla13_RSS/CARLA_0.9.13_RSS/PythonAPI/SCSG/SCSG_test/5_i-1_0_c_l_f_1_0'
        try:
            for root, _, files in os.walk(scenario_root + '/filter/'):
                for name in files:
                    f = open(scenario_root + '/filter/' + name, 'r')
                    blueprint = f.readlines()[0]
                    name = name.strip('.txt')
                    f.close()
                    # print(name)
                    # print(blueprint)
                    if name == 'player':
                        self.ego_car_ = AgentData('ego', blueprint)
                    else:
                        self.tps_[name] = AgentData('tp', blueprint)

            # print(self.ego_car_)
            # print(self.tps_)
        except:
            print("no scenario file")
            return # exit self.load()
        
        # set up agent dynamics
        self.ego_car_.set_transform(read_transform(
            os.path.join(scenario_root, 'transform', 'player.npy')))
        self.ego_car_.set_velocity(read_velocity(
            os.path.join(scenario_root, 'velocity', 'player.npy')))
        
        for actor_id, _ in self.tps_.items():
            self.tps_[actor_id].set_transform(read_transform(
                os.path.join(scenario_root, 'transform', actor_id + '.npy')))
            self.tps_[actor_id].set_velocity(read_velocity(
                os.path.join(scenario_root, 'velocity', actor_id + '.npy')))
            if 'pedestrian' in self.tps_[actor_id].blueprint_:
                self.tps_[actor_id].set_ped_control(read_ped_control(
                    os.path.join(scenario_root, 'ped_control', actor_id + '.npy')))
