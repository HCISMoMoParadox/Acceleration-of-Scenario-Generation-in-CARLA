import os
import json


if __name__ == '__main__':
    # set collision dataset root and the maximum frame span
    path = '/run/user/1002/gvfs/smb-share:server=hcis_nas.local,share=carla/Final_Dataset/dataset/collision'
    max_frame = 200

    collision_frame_GT = dict()
    # Since the os.walk will take all filefoler in a path(deeper than 1 layer), we will need a list to filter these folders.
    extra_folder_list = [
        'trajectory_frame',
        'depth',
        'rgb',
        'ray_cast',
        'map',
        'optical_flow',
        'dvs',
        'instance_segmentation'
        ]
    # Traverse filefolder of collision scenarios
    # {scenario_id}/variant_scenario/{weather}/collision_hisotry.json

    for root, dirs, files in os.walk(path):
        for scenario_id in dirs:
            if scenario_id not in extra_folder_list:
                for root, dirs, files in os.walk(os.path.join(path, scenario_id, 'variant_scenario')):
                    for weather in dirs:
                        if weather not in extra_folder_list:
                            try:
                                with open(os.path.join(path, scenario_id, 'variant_scenario', weather, 'collision_history.json')) as collision_file:
                                    collsion_history = json.load(collision_file)
                                    # print(os.path.join(root, weather))
                                    # print(collsion_history[0])
                                    first_collision_frame = collsion_history[0]['frame']
                                    first_collision_id = collsion_history[0]['actor_id']
                                    # print(first_collision_frame)
                                    # print(first_collision_id)

                                    if scenario_id in collision_frame_GT:
                                        collision_frame_GT[scenario_id][weather] = {first_collision_frame: first_collision_id}
                                    else:
                                        collision_frame_GT[scenario_id] = {weather:{first_collision_frame: first_collision_id}}

                            except FileNotFoundError:
                                if 'FileNotFound' in collision_frame_GT:
                                    collision_frame_GT['FileNotFound'].append(scenario_id+'_'+weather)
                                else:
                                    collision_frame_GT['FileNotFound'] = list()
                            
                        # finally:
                        #     pass

with open("collsion_GT.json", "w") as file:
    json.dump(collision_frame_GT, file, indent=4, sort_keys=True)