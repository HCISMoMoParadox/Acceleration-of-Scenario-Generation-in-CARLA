import os
import csv
import json
import matplotlib.pyplot as plt
import re
import math

if __name__ == '__main__':
    path = '/run/user/1002/gvfs/smb-share:server=hcis_nas.local,share=carla/dataset/obstacle'
    max_frame = 200

    # init. for TTC
    TTC_X = []
    TTC_Y = []
    TTC_Y_all = []
    for i in range(max_frame):
        TTC_X.append(i)
        TTC_Y.append(0)
        TTC_Y_all.append(0)

    index_error = 0
    file_not_found = 0 

    for root, dirs1, files in os.walk(path):
        # for dir in dirs, files in os.walk(os.path.join(path, dirs))
        # print (dirs)
        for dir1 in dirs1:
            # print(os.path.join(path, dir1, 'variant_scenario'))
            for root, dirs2, files in os.walk(os.path.join(path, dir1, 'variant_scenario')):
                # print(root)

                for dir2 in dirs2:
                    # with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'rss_predictoins.json')) as rss_file:
                    #     rss_data = json.load(rss_file)
                    #     print(rss_data)
                    # try:
                        # with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'rss_predictrions.json')) as rss_file:
                        #     rss_data = json.load(rss_file)
                        #     print(rss_data)

                        


                    with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'ego_data.json')) as ego_file:
                        ego_data = json.load(ego_file)
                        # read initial position
                        print(dir1)
                        for frame, data in ego_data.items():
                            print('frame:', frame)
                            ego_init_x = data['transform']['x']
                            ego_init_y = data['transform']['y']
                            break

                        with open(os.path.join(path, dir1, 'obstacle', 'obstacle_list.txt')) as obstacle_txt_file:
                            min_dis = 99999999
                            nearest_x = 9999
                            nearest_y = 9999
                            for line in obstacle_txt_file.readlines():
                                # parse 
                                # print(line)
                                tokens = re.split('=|,', line)
                                # tokens = line.split('Transform(Location(x=', ', y=', ', z=')
                                # print(tokens[1], tokens[3])
                                obstacle_x = float(tokens[1])
                                obstacle_y = float(tokens[3])
                                dis = ((ego_init_x-obstacle_x)**2 + (ego_init_y-obstacle_y)**2)**0.5
                                if dis < min_dis:
                                    min_dis = dis
                                    nearest_x = obstacle_x
                                    nearest_y = obstacle_y
                            obstacle_txt_file.close

                            # find the closest-to-obtacle frame
                            nearest_frame = -1
                            min_dis = 99999999
                            for frame, data in ego_data.items():
                                # print(data)
                                if 'transform' in data:
                                    ego_x = data['transform']['x']
                                    ego_y = data['transform']['y']
                                    dis_this_frame = ((nearest_x - ego_x))**2 - ((nearest_y - ego_y))**2
                                    if dis_this_frame < min_dis:
                                        min_dis = dis_this_frame
                                        nearest_frame = int(frame)

                            # count TTC
                            TTC_X = []
                            TTC_Y = []
                            for i in range(max_frame):
                                TTC_X.append(i)
                                TTC_Y.append(0)

                            print('nearset_frame:', nearest_frame)
                            for frame, data in ego_data.items():
                                if 'transform' in data:
                                    ego_x = data['transform']['x']
                                    ego_y = data['transform']['y']
                                    ego_vx = data['speed']['x']
                                    ego_vy = data['speed']['y']
                                    TTC_this_frame = (((nearest_x - ego_x)/ego_vx)**2 + ((nearest_y - ego_y)/ego_vy)**2)**0.5
                                    print(frame, TTC_this_frame)
                                    if nearest_frame - int(frame) >= 0:
                                        TTC_Y[nearest_frame - int(frame)] = math,log(TTC_this_frame,10)
                                        # if TTC_this_frame >= 5:
                                        #     TTC_Y[nearest_frame - int(frame)] = 10
                                        # else:
                                        #     TTC_Y[nearest_frame - int(frame)] = TTC_this_frame
                                    else:
                                        break
                            plt.plot(TTC_X, TTC_Y)
                            plt.xlabel("Before-Closest-Time, scenario id: " + dir1)
                            plt.ylabel("TTC")
                            plt.show()

                            # start to count TTC frame

                            # exit()



                            
    #                 except IndexError:
    #                     index_error = index_error + 1
    #                 except FileNotFoundError:
    #                     file_not_found = file_not_found + 1
    #                 finally:
    #                     pass

    
    print('index_error: ', index_error)
    print('file_notf_found: ', file_not_found)
    plt.plot(TTC_X, TTC_Y)
    plt.plot(TTC_X, TTC_Y_all)
    plt.legend(['TTC to the other actor', 'TTC to all'], loc='best')
    plt.show()