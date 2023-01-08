#!/bin/sh

SERVICE="CarlaUE4"
declare -i RESET_CNT

ulimit -n 4096



# sudo docker run --privileged -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.13 /bin/bash ./CarlaUE4.sh -carla-port=$1 -p $1:$1 & sleep 15	
../../CarlaUE4.sh -carla-port=$1 -p $1:$1 & sleep 15
python ../util/config.py -m Town05 --no-rendering -p $1
# python ../util/config.py -m Town03 --no-rendering -p $1
#python3 -m cProfile -o pi2.pstats scenario_initializer.py
python scenario_worker.py -p $1


# while true
# do
#     # if pgrep "$SERVICE" >/dev/null
#     # then
#     #     echo "$SERVICE is running"
#     # else
#     #     echo "$SERVICE is  stopped"
#     #     reset_count += 1
#     echo "Reset count: $RESET_CNT"
#     sudo docker run --privileged --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.13 /bin/bash ./CarlaUE4.sh -carla-port=$1 & sleep 15	
#     # python ../util/config.py -m Town05 --no-rendering
#     python ../util/config.py -m Town03 --no-rendering
#     #python3 -m cProfile -o pi2.pstats scenario_initializer.py
#     python scenario_worker.py
#     # fi
#     # sleep 30
# done

#python3 -m cProfile -o pi3.pstats ce_test.py
#python -m gprof2dot -f pstats pi3.pstats | dot -T png -o pi3_profile.png