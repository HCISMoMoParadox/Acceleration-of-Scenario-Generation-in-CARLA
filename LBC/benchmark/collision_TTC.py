import os
import json
import matplotlib.pyplot as plt
import metric

if __name__ == '__main__':
    path = '/run/user/1002/gvfs/smb-share:server=hcis_nas.local,share=carla/DATA/collision'
    max_frame = 200

    # init. for TTC
    TTC_X = []
    TTC_Y = []
    TTC_Y_all = []
    for i in range(max_frame):
        TTC_X.append(i)
        TTC_Y.append(0)
        TTC_Y_all.append(0)

    for root, dirs, files in os.walk(path):
        # for dir in dirs, files in os.walk(os.path.join(path, dirs))
        # print (dirs)
        for dir1 in dirs:
            # print(os.path.join(path, dir1, 'variant_scenario'))
            for root, dirs, files in os.walk(os.path.join(path, dir1, 'variant_scenario')):
                # print(root)
                for dir2 in dirs:
                    # with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'rss_predictoins.json')) as rss_file:
                    #     rss_data = json.load(rss_file)
                    #     print(rss_data)
                    try:
                        # with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'rss_predictrions.json')) as rss_file:
                        #     rss_data = json.load(rss_file)
                        #     print(rss_data)

                        with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'collision_history.json')) as collision_file:
                            collsion_history = json.load(collision_file)
                            print(os.path.join(root, dir2))
                            # print(collsion_history[0])
                            first_collision_frame = collsion_history[0]['frame']
                            first_collision_id = collsion_history[0]['actor_id']
                            print(first_collision_frame)
                            print(first_collision_id)

                            with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'rss_predictoins.json')) as rss_file:
                                rss_data = json.load(rss_file)
                                # print(rss_data)
                                for frame, data in rss_data.items():
                                    # print(type(frame))
                                    # print(data['EgoCarIsSafe'])
                                    frame_number = int(frame)
                                    if frame_number > first_collision_frame:
                                        break
                                    
                                    if not data['EgoCarIsSafe']:# and (first_collision_id in data['DangerousIds']):
                                        # print(first_collision_id in data['DangerousIds'])
                                        if first_collision_id in data['DangerousIds']:
                                            TTC_Y[first_collision_frame-frame_number] += 1    
                                        # print(frame)
                                        TTC_Y_all[first_collision_frame-frame_number] += 1
                            
                            # exit()

                    except:
                        pass
                    finally:
                        pass
    plt.plot(TTC_X, TTC_Y)
    plt.plot(TTC_X, TTC_Y_all)
    plt.legend(['TTC to the collided actor', 'TTC to all'], loc='best')
    plt.show()

                