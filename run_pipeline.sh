##!/bin/bash
#for value in {1..1}
#do
#python train.py --test_subject $value
#python test.py --test_subject $value
#done

#!/bin/bash

for value in UBC_easy UBC_medium UBC_hard
do

if [ $value = UBC_easy ]
then
    python train.py --dataset_name $value --steps 25000
    cp -r /media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_easy/train /media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_medium/
elif [ $value = UBC_medium ]
then
    python train.py --dataset_name $value --steps 50000
    cp -r /media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_medium/train /media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_hard/
elif [ $value = UBC_hard ]
then
    python train.py --dataset_name $value --steps 65000
fi

done