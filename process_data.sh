#!/bin/sh

echo "Input the scenario_type you want to process"
echo "Choose from the following options:"
echo "1 - collision"
echo "2 - obstacle"
echo "3 - interactive"
echo "4 - non-interactive"
read -p "Enter scenario type ID to create a data video: " ds_id

scenario_type="interactive"
if [ ${ds_id} == 1 ]
then
    scenario_type="collision"
elif [ ${ds_id} == 2 ]
then
    scenario_type="obstacle"
elif [ ${ds_id} == 3 ]
then
    scenario_type="interactive"
elif [ ${ds_id} == 4 ]
then
    scenario_type="non-interactive"
else
    echo "Invalid ID!!!"
    echo "run default setting : interactive"
fi

len=${#scenario_type}
len=$((len + 19))

folder=`ls -d ./data_collection/${scenario_type}/*`

for scenario_name in $folder
do
	# mv ./data_collection/${scenario_type}/${scenario_name:$len}/${scenario_name:$len}.mp4 ./data_collection/${scenario_type}/${scenario_name:$len}/${weather[${w[${i}]}]}_${random_actor[j]}_

	x="./data_collection/${scenario_type}/${scenario_name:$len}/variant_scenario"
	f=`ls -d ${x}/*_*_`

	for name in $f
	do 
		echo ${name}
		python make_video.py --path $name
	done
	./zip_data.sh ${scenario_type} ${scenario_name:$len} &	
done






















