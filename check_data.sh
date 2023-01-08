#!/bin/bash
# This shell is going to create the video for each scenario_ID

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

mkdir "./data_collection/${scenario_type}_fail/"
for eachfile in $folder
do
    echo "$eachfile/variant_scenario"
    check=`ls $eachfile/variant_scenario | wc -l`
	if [ ${check} -lt 4 ]
	then
		mv $eachfile ./data_collection/${scenario_type}_fail/
    fi
done
