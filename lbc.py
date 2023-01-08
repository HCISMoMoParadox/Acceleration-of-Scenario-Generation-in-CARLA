#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


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


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

# LBC
import torch
from dataset import CarlaDataset
from converter import Converter
from map_model import MapModel
import common
import pathlib
import uuid
import copy
from collections import deque
# from controller import VehiclePIDController
import cv2
# LBC

'''
modules to be install:

imgaug==0.4.0
pandas==0.25.3
pytz==2019.3
pytorch-lightning==0.7.1
wandb==0.8.29
urllib3==1.25.8
idna==2.9
'''

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            # model 3
            blueprint_library = self.world.get_blueprint_library()
            model_3 = blueprint_library.filter("model3")[0]
            # model 3
            self.player = self.world.try_spawn_actor(model_3, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            blueprint_library = self.world.get_blueprint_library()
            model_3 = blueprint_library.filter("model3")[0]
            self.player = self.world.try_spawn_actor(model_3, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        self.frame = None
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
        self.frame = sensor_data.frame


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.sensor_lbc_seg = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.bev_map = None
        self.bev_map_frame = None
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid),
                # LBC top view
                (carla.Transform(carla.Location(x=0, y=0,
                 z=100.0), carla.Rotation(pitch=-90.0)), Attachment.Rigid)
                ]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.SpringArm)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        self.bev_bp = bp_library.find('sensor.camera.rgb')
        self.bev_bp.set_attribute('image_size_x', str(512))
        self.bev_bp.set_attribute('image_size_y', str(512))
        self.bev_bp.set_attribute('fov', str(50.0))
        # if self.bev_bp.has_attribute('gamma'):
        #     self.bev_bp.set_attribute('gamma', str(gamma_correction))


        self.bev_seg_bp = bp_library.find('sensor.camera.semantic_segmentation')
        self.bev_seg_bp.set_attribute('image_size_x', str(512))
        self.bev_seg_bp.set_attribute('image_size_y', str(512))
        self.bev_seg_bp.set_attribute('fov', str(50.0))

        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            if self.sensor_lbc_seg is not None:
                self.sensor_lbc_seg.destroy()
                self.surface = None
            self.sensor_lbc_seg = self._parent.get_world().spawn_actor(
                    self.bev_seg_bp,
                    self._camera_transforms[5][0],
                    attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

            self.sensor_lbc_seg.listen(lambda image: CameraManager._parse_image(
                    weak_self, image, 'lbc_seg'))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, view='non-top'):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        elif view == 'lbc_seg':
            def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
                result = list()

                transform = vehicle.get_transform()
                pos = transform.location
                theta = np.radians(90 + transform.rotation.yaw)
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)],
                    ])

                for light in lights:
                    delta = light.get_transform().location - pos

                    target = R.T.dot([delta.x, delta.y])
                    target *= pixels_per_meter
                    target += size // 2

                    if min(target) < 0 or max(target) >= size:
                        continue

                    trigger = light.trigger_volume
                    light.get_transform().transform(trigger.location)
                    dist = trigger.location.distance(vehicle.get_location())
                    a = np.sqrt(
                            trigger.extent.x ** 2 +
                            trigger.extent.y ** 2 +
                            trigger.extent.z ** 2)
                    b = np.sqrt(
                            vehicle.bounding_box.extent.x ** 2 +
                            vehicle.bounding_box.extent.y ** 2 +
                            vehicle.bounding_box.extent.z ** 2)

                    if dist > a + b:
                        continue

                    result.append(light)

                return result
            def draw_traffic_lights(image, vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
                from PIL import Image, ImageDraw
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)

                transform = vehicle.get_transform()
                pos = transform.location
                theta = np.radians(90 + transform.rotation.yaw)
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)],
                    ])

                for light in lights:
                    delta = light.get_transform().location - pos

                    target = R.T.dot([delta.x, delta.y])
                    target *= pixels_per_meter
                    target += size // 2

                    if min(target) < 0 or max(target) >= size:
                        continue

                    trigger = light.trigger_volume
                    light.get_transform().transform(trigger.location)
                    dist = trigger.location.distance(vehicle.get_location())
                    a = np.sqrt(
                            trigger.extent.x ** 2 +
                            trigger.extent.y ** 2 +
                            trigger.extent.z ** 2)
                    b = np.sqrt(
                            vehicle.bounding_box.extent.x ** 2 +
                            vehicle.bounding_box.extent.y ** 2 +
                            vehicle.bounding_box.extent.z ** 2)

                    if dist > a + b:
                        continue

                    x, y = target
                    draw.ellipse(
                            (x-radius, y-radius, x+radius, y+radius),
                            23 + light.state.real)

                return np.array(image)
            '''
            Instance Variables: image
                fov (float - degrees): Horizontal field of view of the image.
                height (int): Image height in pixels.
                width (int): Image width in pixels.
                raw_data (bytes)
            '''
            # print('fov:', image.fov)
            # print('height:', image.height)
            # print('width:', image.width)
            # print('raw_data:', type(image.raw_data))
            # print(image.raw_data[10])
            # exit()
            # pass
            
            # take out the raw data
            array_raw = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            # print("array.shape", array.shape) # array.shape (1048576,)
            array_raw = np.reshape(array_raw, (image.height, image.width, 4))
            actors = self._parent.get_world().get_actors()
            traffic_lights = get_nearby_lights(self._parent, actors.filter('*traffic_light*'))
            array_raw = draw_traffic_lights(array_raw[:, :, 2], self._parent, traffic_lights)
            # cv2.imwrite
            # print(array[256, 256, 0], array[256, 256, 1], array[256, 256, 2], array[256, 256, 3]) # 0 0 10 255 -> channel 2 as raw data -> array = array[:, :, 2]
            self.bev_map = copy.deepcopy(array_raw)
            self.bev_map_frame = image.frame
            # print(image.timestamp)

            # self.surface = pygame.surfarray.make_surface(array_raw.swapaxes(0, 1))  # show bev map on pygame canvas

            # print(type(self.bev_map))
            # print('in_call_back:',self.bev_map)
            # print(self.bev_map.shape)

            # show on screen
            # image.convert(self.sensors[5][1])
            # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            # array = np.reshape(array, (image.height, image.width, 4))
            # array = array[:, :, :3]
            # array = array[:, :, ::-1]
            # self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
        else:
            # pass
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        # import time
        # if args.map is not None:    # change map
        #     client2 = carla.Client(args.host, args.port, worker_threads=1)
        #     client2.set_timeout(10.0)
        #     print('load map %r.' % args.map)
        #     client2.load_world(args.map)
        #     # time.sleep(10)
        # # client2 = carla.Client(args.host, args.port, worker_threads=1)
        # # client2.set_timeout(10.0)
        # client2.reload_world()   # reload map for cleaning memory
        # time.sleep(20)
        



        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        # sync mode
        args.sync = True

        sim_world = client.get_world()


        # import time
        # client.reload_world()   # reload map for cleaning memory
        # time.sleep(10)
        # if args.map is not None:    # change map
        #     client2 = carla.Client(args.host, args.port, worker_threads=1)
        #     client2.set_timeout(10.0)
        #     print('load map %r.' % args.map)
        #     client2.load_world(args.map)
        #     time.sleep(10)

        
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        # get frame number
        frame = None
        topdown = None
        N_CLASSES = len(common.COLOR)

        if args.sync:
            frame = sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()


        # get waypoints
        map = sim_world.get_map()
        def find_next_target(max_distance, wp_now):
            wp_last = None
            wp_next = wp_now.next(2)[0] # next waypoint in 2 meters
            # print(wp_next, len(wp_next))
            while wp_now.transform.location.distance(wp_next.transform.location) < max_distance:
                # print(wp_now.transform.location.distance(wp_next.transform.location))
                wp_last = wp_next
                wp_next = wp_next.next(2)[0]
                # if len(wp_next) > 1:
                #     wp_next = wp_next[0]
            # print(wp_now.transform.location.distance(wp_next.transform.location))
            return wp_last
            
        min_distance = 7.5 #7.5
        max_distance = 25.0
        wp_now = map.get_waypoint(location=world.player.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        target = find_next_target(max_distance, wp_now)
        # sim_world.debug.draw_point(target.transform.location, 0.1, carla.Color(255,0,0), 0.0, True)
        # print(target)

        # print(frame)
        
        # controller
        # ego_control = VehiclePIDController(world.player, args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
        #                                                             max_throttle=1.0, max_brake=1.0, max_steering=1.0)
        DEBUG = False


        class PIDController(object):
            def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
                self._K_P = K_P
                self._K_I = K_I
                self._K_D = K_D

                self._window = deque([0 for _ in range(n)], maxlen=n)
                self._max = 0.0
                self._min = 0.0

            def step(self, error):
                self._window.append(error)
                self._max = max(self._max, abs(error))
                self._min = -abs(self._max)

                if len(self._window) >= 2:
                    integral = np.mean(self._window)
                    derivative = (self._window[-1] - self._window[-2])
                else:
                    integral = 0.0
                    derivative = 0.0

                if DEBUG:
                    import cv2

                    canvas = np.ones((100, 100, 3), dtype=np.uint8)
                    w = int(canvas.shape[1] / len(self._window))
                    h = 99

                    for i in range(1, len(self._window)):
                        y1 = (self._max - self._window[i-1]) / (self._max - self._min + 1e-8)
                        y2 = (self._max - self._window[i]) / (self._max - self._min + 1e-8)

                        cv2.line(
                                canvas,
                                ((i-1) * w, int(y1 * h)),
                                ((i) * w, int(y2 * h)),
                                (255, 255, 255), 2)

                    canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))

                    cv2.imshow('%.3f %.3f %.3f' % (self._K_P, self._K_I, self._K_D), canvas)
                    cv2.waitKey(1)

                return self._K_P * error + self._K_I * integral + self._K_D * derivative

        ego_control = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # load model
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_epochs', type=int, default=50)
        parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
        parser.add_argument('--id', type=str, default=uuid.uuid4().hex)

        parser.add_argument('--heatmap_radius', type=int, default=5)
        parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
        parser.add_argument('--command_coefficient', type=float, default=0.1)
        parser.add_argument('--temperature', type=float, default=10.0)
        parser.add_argument('--hack', action='store_true', default=False)

        # Data args.
        parser.add_argument('--dataset_dir', type=pathlib.Path, default='/home/momoparadox/LBC/2020_CARLA_challenge/sample_data/route_00')
        parser.add_argument('--batch_size', type=int, default=32)

        # Optimizer args.
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)

        parsed = parser.parse_args()
        # parsed.save_dir = parsed.save_dir / parsed.id
        # parsed.save_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = MapModel(parsed)
        
        net = MapModel.load_from_checkpoint('/home/momoparadox/LBC/epoch=34_01.ckpt')

        net.cuda()

        net.eval()
        
        frame_step = 5
        last_inf_frame = frame - frame_step

        
        
        # spawn actors
        def get_actor_blueprints(world, filter, generation):
            bps = world.get_blueprint_library().filter(filter)

            if generation.lower() == "all":
                return bps

            # If the filter returns only one bp, we assume that this one needed
            # and therefore, we ignore the generation
            if len(bps) == 1:
                return bps

            try:
                int_generation = int(generation)
                # Check if generation is in available generations
                if int_generation in [1, 2]:
                    bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                    return bps
                else:
                    print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                    return []
            except:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        number_of_vehicles = 50

        blueprints = get_actor_blueprints(sim_world, args.filter, 'all')
        # blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if True: # safe actors
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = sim_world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        # hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            # if hero:
            #     blueprint.set_attribute('role_name', 'hero')
            #     hero = False
            # else:
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        vehicles_list = []
        synchronous_master = False # for sync mode = false
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # write bev_as_video
        out = cv2.VideoWriter("test_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,  (256, 256) )
        while True:
            # loop for LBC
            
            # get input
            if last_inf_frame == frame - frame_step:
                last_inf_frame = frame
            # if True:
                # update waypoint
                wp_now = map.get_waypoint(location=world.player.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp_now.transform.location.distance(target.transform.location) < min_distance:
                    target = find_next_target(max_distance, wp_now)
                    # sim_world.debug.draw_point(target.transform.location, 0.1, carla.Color(255,0,0), 0.0, True)

                
                # wait for bev map
                while True:    
                    if world.camera_manager.bev_map_frame == frame:
                    # if True:
                        topdown = world.camera_manager.bev_map
                        # print('in_loop:', topdown)
                        break
                # wait for compass data(orientation)
                while True:    
                    if world.imu_sensor.frame == frame:
                    # if True:
                        theta = world.imu_sensor.compass
                        break
                print(frame)
                # print(topdown.timestamp)
                theta = theta/180.0 * np.pi# + np.pi / 2
                
                # print(theta)
                R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
                ])

                PIXELS_PER_WORLD = 5.5 # from coverter.py 
                ego_location = world.player.get_location()
                ego_xy = np.float32([ego_location.x, ego_location.y])
                print(target)
                target_loc = target.transform.location
                target_xy = np.float32([target_loc.x, target_loc.y])
                target_xy = R.T.dot(target_xy - ego_xy)
                target_xy_print = target_xy
                target_xy *= PIXELS_PER_WORLD
                # target_xy_print = target_xy
                target_xy += [128, 256]
                target_xy = np.clip(target_xy, 0, 256)
                target_xy = torch.FloatTensor(target_xy)

                print('target', target_xy)

                
                
                print('topdown.shape',topdown.shape)
                # topdown = topdown.crop((128, 0, 128 + 256, 256))
                topdown = topdown[0: 256, 128:384]
                # top_down_to_draw = copy.deepcopy(topdown)
                # print(topdown.shape)
                # print(topdown)
                
                topdown = np.array(topdown)
                topdown = common.CONVERTER[topdown]

                top_down_to_draw = copy.deepcopy(topdown) # pass to draw

                topdown = torch.LongTensor(topdown)
                topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()
                # inference
                points_pred = net.forward(topdown, target_xy)
                print(points_pred)
                control = net.controller.forward(points_pred)
                print(control)
                

                # out, (target_heatmap,) = net.forward(topdown, target, debug=True)

                # alpha = torch.rand(out.shape).type_as(out)
                # between = alpha * out + (1-alpha) * points
                # out_cmd = net.controller.forward(between)
                        

                # alpha = torch.rand(out.shape).type_as(out)
                # between = alpha * out + (1-alpha) * points

                # draw points
                points_to_draw = points_pred.cpu().data.numpy()
                points_to_draw = points_to_draw * 256
                print('points_to_draw:', points_to_draw)

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
                print('pred_points', points_draw)
                # sim_world.debug.draw_point(carla.Location(x=points_draw[0][0], y=points_draw[0][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
                # sim_world.debug.draw_point(carla.Location(x=points_draw[1][0], y=points_draw[1][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
                # sim_world.debug.draw_point(carla.Location(x=points_draw[2][0], y=points_draw[2][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
                # sim_world.debug.draw_point(carla.Location(x=points_draw[3][0], y=points_draw[3][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
                
                
                save_canvas = np.zeros((256, 256, 3), np.int8)
                print('points_to_draw', points_to_draw)
                save_canvas = common.COLOR[top_down_to_draw]
                for xy in points_to_draw[0]:
                    print(xy)
                    x_int = 128 + int(xy[0])
                    y_int = int(xy[1])
                    save_canvas[y_int][x_int] = [255, 255, 255]

                    x_target = int(target_xy_print[0])
                    y_target = int(target_xy_print[1])
                    save_canvas[y_target][x_target] = [0, 255, 0]
                    for pad in range(1, 3):
                        save_canvas[y_int-pad][x_int+pad] = [255, 255, 255]
                        save_canvas[y_int+pad][x_int+pad] = [255, 255, 255]
                        save_canvas[y_int-pad][x_int-pad] = [255, 255, 255]
                        save_canvas[y_int+pad][x_int-pad] = [255, 255, 255]

                        save_canvas[y_target+pad][x_target+pad] = [0, 255, 0]
                        save_canvas[y_target+pad][x_target-pad] = [0, 255, 0]
                        save_canvas[y_target-pad][x_target+pad] = [0, 255, 0]
                        save_canvas[y_target-pad][x_target-pad] = [0, 255, 0]

                    # print(common.COLOR[top_down_to_draw[x_int][y_int]])
                # cv2.imwrite('./test_result/'+str(frame)+'.png', cv2.cvtColor(save_canvas, cv2.COLOR_RGB2BGR))
                out.write(cv2.cvtColor(save_canvas, cv2.COLOR_RGB2BGR))


                # control
                control = control.cpu().data.numpy()
                print(control)
                steer = control[0][0] #* 1.4
                desired_speed = control[0][1]
                print(steer, desired_speed)
                speed = world.player.get_velocity().length()

                brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1
                # brake = desired_speed < 0.4*1.4 or (speed / desired_speed) > 1.1 *1.4

                delta = np.clip(desired_speed - speed, 0.0, 0.25)
                throttle = ego_control.step(delta)
                throttle = np.clip(throttle, 0.0, 0.75)
                throttle = throttle if not brake else 0.0

                control = carla.VehicleControl()
                control.steer = float(steer)
                control.throttle = float(throttle)
                control.brake = float(brake)
            
            # else:
            #     # draw points
            #     points_to_draw = points_pred.cpu().data.numpy()
            #     points_to_draw = points_to_draw * 256
            #     print('points_to_draw:', points_to_draw)

            #     # points_to_draw -= [128, 256]
            #     # points_to_draw /= PIXELS_PER_WORLD
            #     # print(points_to_draw)
            #     R = np.array([
            #     [np.cos(-theta), -np.sin(-theta)],
            #     [np.sin(-theta),  np.cos(-theta)],
            #     ])
            #     # print('R', R)
            #     # points_to_draw = (1/R.T).dot(points_to_draw[0], axis=1)
            #     points_draw = []
            #     points_draw.append(R.T.dot(points_to_draw[0][0]))
            #     points_draw.append(R.T.dot(points_to_draw[0][1]))
            #     points_draw.append(R.T.dot(points_to_draw[0][2]))
            #     points_draw.append(R.T.dot(points_to_draw[0][3]))
            #     print('pred_points', points_draw)
            #     # sim_world.debug.draw_point(carla.Location(x=points_draw[0][0], y=points_draw[0][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
            #     # sim_world.debug.draw_point(carla.Location(x=points_draw[1][0], y=points_draw[1][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
            #     # sim_world.debug.draw_point(carla.Location(x=points_draw[2][0], y=points_draw[2][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
            #     # sim_world.debug.draw_point(carla.Location(x=points_draw[3][0], y=points_draw[3][1], z=0.5), 0.1, carla.Color(0,255,0), 0.0, True)
                
                
            #     save_canvas = np.zeros((256, 256, 3), np.int8)
            #     print('points_to_draw', points_to_draw)
            #     save_canvas = common.COLOR[top_down_to_draw]
            #     for xy in points_to_draw[0]:
            #         print(xy)
            #         x_int = 128 + int(xy[0])
            #         y_int = int(xy[1])
            #         save_canvas[y_int][x_int] = [255, 255, 255]

            #         x_target = int(target_xy_print[0])
            #         y_target = int(target_xy_print[1])
            #         save_canvas[y_target][x_target] = [0, 255, 0]
            #         for pad in range(1, 3):
            #             save_canvas[y_int-pad][x_int+pad] = [255, 255, 255]
            #             save_canvas[y_int+pad][x_int+pad] = [255, 255, 255]
            #             save_canvas[y_int-pad][x_int-pad] = [255, 255, 255]
            #             save_canvas[y_int+pad][x_int-pad] = [255, 255, 255]
                        
            #             save_canvas[y_target+pad][x_target+pad] = [0, 255, 0]
            #             save_canvas[y_target+pad][x_target-pad] = [0, 255, 0]
            #             save_canvas[y_target-pad][x_target+pad] = [0, 255, 0]
            #             save_canvas[y_target-pad][x_target-pad] = [0, 255, 0]
            #         # print(common.COLOR[top_down_to_draw[x_int][y_int]])
            #     # cv2.imwrite('./test_result/'+str(frame)+'.png', cv2.cvtColor(save_canvas, cv2.COLOR_RGB2BGR))
            #     out.write(cv2.cvtColor(save_canvas, cv2.COLOR_RGB2BGR))
                
            # always keep control
            world.player.apply_control(control)


            # tick the world
            if args.sync:
                frame = sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        out.release()

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',  # 512x512 for map model
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '-m', '--map',
        help='load a new map, use --list to see available maps')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
