import pathlib
import uuid
from collections import deque
import argparse
import carla
import weakref
import math
import numpy as np

def update_target(agent):
    min_distance = 7.5
    max_distance = 25

    waypoint_now = agent.actor_.get_location()
    if waypoint_now.distance(agent.target_.location) < min_distance:
        if agent.index_ == len(agent.data_.transform_):
            return agent.data_.transform_[-1]

        while (agent.index_ < len(agent.data_.transform_) and
                waypoint_now.distance(agent.data_.transform_[agent.index_].location) < max_distance):
            agent.index_ += 1

    agent.target_ = agent.data_.transform_[agent.index_-1]

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

class LBCController(object):
    '''
    A PID controller for LBC
    '''
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        # if DEBUG:
        #     import cv2

        #     canvas = np.ones((100, 100, 3), dtype=np.uint8)
        #     w = int(canvas.shape[1] / len(self._window))
        #     h = 99

        #     for i in range(1, len(self._window)):
        #         y1 = (self._max - self._window[i-1]) / (self._max - self._min + 1e-8)
        #         y2 = (self._max - self._window[i]) / (self._max - self._min + 1e-8)

        #         cv2.line(
        #                 canvas,
        #                 ((i-1) * w, int(y1 * h)),
        #                 ((i) * w, int(y2 * h)),
        #                 (255, 255, 255), 2)

        #     canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))

        #     cv2.imshow('%.3f %.3f %.3f' % (self._K_P, self._K_I, self._K_D), canvas)
        #     cv2.waitKey(1)

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


def LBCArgument():
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
    parser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port of CARLA Simulator (default: 2000)')


    parsed = parser.parse_args()
    # parsed.save_dir = parsed.save_dir / parsed.id
    # parsed.save_dir.mkdir(parents=True, exist_ok=True)

    return parsed

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

class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.other_actor_id = 0 # init as 0 for static object
        self.wrong_collision = False
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))
        self.collision = False

    # def get_collision_history(self):
    #     history = collections.defaultdict(int)
    #     for frame, intensity in self.history:
    #         history[frame] += intensity
    #     return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.history.append({'frame': event.frame, 'actor_id': event.other_actor.id})
