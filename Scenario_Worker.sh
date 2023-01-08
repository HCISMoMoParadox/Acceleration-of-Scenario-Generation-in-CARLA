#!/bin/sh

SERVICE="CarlaUE4"
declare -i RESET_CNT

ulimit -n 4096

while true
do
    if pgrep "$SERVICE" >/dev/null
    then
        echo "$SERVICE is running"
    else
        echo "$SERVICE is  stopped"
        reset_count += 1
        echo "Reset count: $RESET_CNT"
        ../../CarlaUE4.sh & sleep 15	
        python ../util/config.py -m Town05 --no-rendering
        # python ../util/config.py -m Town03 --no-rendering
        #python3 -m cProfile -o pi2.pstats scenario_initializer.py
        python scenario_initializer.py
    fi
    sleep 30
done

#python3 -m cProfile -o pi3.pstats ce_test.py
#python -m gprof2dot -f pstats pi3.pstats | dot -T png -o pi3_profile.png