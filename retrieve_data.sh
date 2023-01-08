echo "Input the nas location"
echo "ex: /run/user/1000/gvfs/smb-share:server=hcis_nas.local,share=carla/mini_set"
read -p "$: " path_to_nas

echo " "
echo "Input the scenario_type of data you want to retrive"
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

echo " "
echo "Do you need depth data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_depth

echo " "
echo "Do you need rgb data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_rgb

echo " "
echo "Do you need lidar data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_lidar

echo " "
echo "Do you need semantic segmentation data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_ss

echo " "
echo "Do you need optical flow data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_of

echo " "
echo "Do you need bbox data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_bb

echo " "
echo "Retrive front data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_front

echo " "
echo "Retrive back data? "
echo "0 - False"
echo "1 - True"
read -p ": " if_back

echo " "
echo "Retrive DVS? "
echo "0 - False"
echo "1 - True"
read -p ": " if_dvs

echo " "
echo "Retrive Learning by cheat top view datas? "
echo "0 - False"
echo "1 - True"
read -p ": " if_lbc


s=`ls -d ${path_to_nas}/${scenario_type}/*`
n="${path_to_nas}/${scenario_type}/"
len_of_path=${#n}
for basic_root in $s
do
    #echo $basic_root
    scenario_id=${basic_root:len_of_path}
    #echo $scenario_id

    output="./data_collection/${scenario_type}/${scenario_id}/" # path to store data
    root=$basic_root"/variant_scenario/"
    len=${#root}
    folder=`ls -d ${root}*`
    
    for eachfile in $folder
    do
        # echo ${eachfile: $len} 
        
        #--- depth -- x6

        if [ ${if_depth} == 1 ]
        then
            if [ ${if_back} == 1 ]
            then 
                file_name="depth_back"
                cp ${eachfile}/depth/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="depth_back_left"
                cp ${eachfile}/depth/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="depth_back_right"
                cp ${eachfile}/depth/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi

            if [ ${if_front} == 1 ]
            then 
                file_name="depth_front"
                cp ${eachfile}/depth/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
                
                file_name="depth_left"
                cp ${eachfile}/depth/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="depth_right"
                cp ${eachfile}/depth/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi
        fi

        # --- dvs 
        if [ ${if_dvs} == 1 ]
        then 

            file_name="dvs"
            cp ${eachfile}/dvs/${file_name}.zip ./
            unzip ./${file_name}.zip
            rm ./${file_name}.zip
        fi

        # --- optical_flow

        if [ ${if_of} == 1 ]
        then  

            file_name="flow"
            cp ${eachfile}/optical_flow/${file_name}.zip ./
            unzip ./${file_name}.zip
            rm ./${file_name}.zip
        fi

        # --- bbox 
        if [ ${if_bb} == 1 ]
        then  
            if [ ${if_back} == 1 ]
            then 
                file_name="back"
                cp ${eachfile}/bbox/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="back_left"
                cp ${eachfile}/bbox/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="back_right"
                cp ${eachfile}/bbox/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi
            if [ ${if_front} == 1 ]
            then
                file_name="front"
                cp ${eachfile}/bbox/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="left"
                cp ${eachfile}/bbox/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="right"
                cp ${eachfile}/bbox/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi

            file_name="top"
            cp ${eachfile}/bbox/${file_name}.zip ./
            unzip ./${file_name}.zip
            rm ./${file_name}.zip
        fi

        # -- ray_cast
        if [ ${if_lidar} == 1 ]
        then  
            file_name="lidar"
            cp ${eachfile}/ray_cast/${file_name}.zip ./
            unzip ./${file_name}.zip
            rm ./${file_name}.zip
        fi


        # --- rgb 
        if [ ${if_rgb} == 1 ]
        then 
            if [ ${if_back} == 1 ]
            then

                file_name="back"
                cp ${eachfile}/rgb/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="back_left"
                cp ${eachfile}/rgb/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="back_right"
                cp ${eachfile}/rgb/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi
            if [ ${if_front} == 1 ]
            then
                file_name="front"
                cp ${eachfile}/rgb/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="left"
                cp ${eachfile}/rgb/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="right"
                cp ${eachfile}/rgb/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi

            file_name="top"
            cp ${eachfile}/rgb/${file_name}.zip ./
            unzip ./${file_name}.zip
            rm ./${file_name}.zip
            
            if [ ${if_lbc} == 1 ]
            then
                file_name="lbc_img"
                cp ${eachfile}/rgb/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi
        fi

        # --- semantic_segmentation 
        if [ ${if_ss} == 1 ]
        then 
            if [ ${if_back} == 1 ]
            then
                file_name="seg_back"
                cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="seg_back_left"
                cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="seg_back_right"
                cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi
            if [ ${if_front} == 1 ]
            then
                file_name="seg_front"
                cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="seg_left"
                cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip

                file_name="seg_right"
                cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi

            file_name="seg_top"
            cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
            unzip ./${file_name}.zip
            rm ./${file_name}.zip
            
            if [ ${if_lbc} == 1 ]
            then
                file_name="lbc_seg"
                cp ${eachfile}/semantic_segmentation/${file_name}.zip ./
                unzip ./${file_name}.zip
                rm ./${file_name}.zip
            fi
        fi
        # -- topology
        file_name="topology"
        cp ${eachfile}/${file_name}.zip ./
        unzip ./${file_name}.zip
        rm ./${file_name}.zip

        # -- trajectory
        file_name="trajectory"
        cp ${eachfile}/${file_name}.zip ./
        unzip ./${file_name}.zip
        rm ./${file_name}.zip

        cp ${eachfile}/$scenario_id.mp4 ${output}variant_scenario/${eachfile: $len}
        cp ${eachfile}/dynamic_description.json ${output}variant_scenario/${eachfile: $len}
        cp ${eachfile}/ego_data.json ${output}variant_scenario/${eachfile: $len}
        cp ${eachfile}/retrieve_gt.txt ${output}variant_scenario/${eachfile: $len}
    done
    cp -r $basic_root/filter ${output}
    cp -r $basic_root/obstacle ${output}
    cp -r $basic_root/ped_control ${output}
    cp -r $basic_root/traffic_light ${output}
    cp -r $basic_root/transform ${output}
    cp -r $basic_root/velocity ${output}
    cp $basic_root/scenario_description.json ${output}
    cp $basic_root/timestamp.txt ${output}
    cp $basic_root/sample.mp4 ${output}
    echo "finishing unzip file"
done

