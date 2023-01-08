import os
import csv
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '/run/user/1002/gvfs/smb-share:server=hcis_nas.local,share=carla/dataset/interactive'
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
                    try:
                        # with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'rss_predictrions.json')) as rss_file:
                        #     rss_data = json.load(rss_file)
                        #     print(rss_data)

                        with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'trajectory_frame', dir1+'.csv')) as f:
                            print('scenario: ', dir1, dir2)
                            csvreader = csv.reader(f)
                            header = next(csvreader)
                            rows = []
                            for row in csvreader:
                                rows.append(row)
                            first_frame = int(rows[0][0])
                            last_frame = int(rows[-1][0])
                            # print(first_frame, last_frame)

                            # find the other actor id: the 2nd least id  (the least id is the ego-car's id)
                            # for the same frame the last id is ego car's, the 2nd last id is the other actor's
                            index = 0
                            while(int(rows[index][0]) == first_frame):
                                index = index + 1
                            
                            ego_car_id = int(rows[index-1][1])
                            other_actor_id = int(rows[index-2][1])

                            # index = 0
                            # ego_car_id = 9999999
                            # while(int(rows[index][0]) == first_frame):
                            #     if rows[index][1] != 'player':
                            #         if int(rows[index][1]) < ego_car_id:
                            #             ego_car_id =  int(rows[index][1])
                            #     index = index + 1
                            
                            # index = 0
                            # other_actor_id = 9999999
                            # while(int(rows[index][0]) == first_frame):
                            #     if rows[index][1] != 'player':
                            #         if int(rows[index][1]) < other_actor_id:
                            #             other_actor_id =  int(rows[index][1])
                            #     index = index + 1


                            ego_xy = []
                            other_actor_xy = []
                            for row in rows:
                                if row[1] == str(other_actor_id):
                                    other_actor_xy.append([row[3], row[4]])
                                elif row[1] == str(ego_car_id):
                                    ego_xy.append([row[3], row[4]])

                            print('other id done')

                            min_dist = 999999999
                            min_dis_frame = -1
                            # find the closet distance and frame_number
                            index = 0
                            for x1, y1 in ego_xy:
                                for x2, y2 in other_actor_xy:
                                    dis = (float(x1) - float(x2))**2 + (float(y1) - float(y2))**2
                                    if dis < min_dist:
                                        min_dist = dis
                                        min_dis_frame = index
                                index = index + 1
                                # print(index)
                            min_dis_frame = min_dis_frame + first_frame

                            print('frame from ', first_frame, ' to ', last_frame)
                            print('min_dis_frame: ', min_dis_frame)
                            print('min_dist: ', min_dist**0.5, 'm')
                            
                            with open(os.path.join(path, dir1, 'variant_scenario', dir2, 'rss_predictoins.json')) as rss_file:
                                rss_data = json.load(rss_file)
                                # print(rss_data)
                                for frame, data in rss_data.items():
                                    # print(type(frame))
                                    # print(data['EgoCarIsSafe'])
                                    frame_number = int(frame)
                                    if frame_number > min_dis_frame:
                                        break
                                    
                                    if not data['EgoCarIsSafe']:# and (first_collision_id in data['DangerousIds']):
                                        # print(first_collision_id in data['DangerousIds'])
                                        if other_actor_id in data['DangerousIds']:
                                            TTC_Y[min_dis_frame-frame_number] += 1    
                                        # print(frame)
                                        TTC_Y_all[min_dis_frame-frame_number] += 1
                    except IndexError:
                        index_error = index_error + 1
                    except FileNotFoundError:
                        file_not_found = file_not_found + 1
                    finally:
                        pass

    
    print('index_error: ', index_error)
    print('file_notf_found: ', file_not_found)
    plt.plot(TTC_X, TTC_Y)
    plt.plot(TTC_X, TTC_Y_all)
    plt.legend(['TTC to the other actor', 'TTC to all'], loc='best')
    plt.show()