TRAIN_KITTI_ORIGINAL(){
cd ..
cd HuggingFace_Trainer/
source_datapath="/data1/KITTI/KITTI_Raw/"
outside_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"
center_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train_SmallSet/"
confidence_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/Reference_PSNR/"
trainlist="/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTISmallSet/kitti_raw_train.txt"
vallist="/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTI_2015_train.txt"
logging_dir="/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/logs"
output_dir="/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/outputs/simple_pamstereo"
tracker_project_name="tracker_project_kitti_raw"

batch_size=2
datathread=4
# visible_list="['left','right','left_left','right_right','center','confidence']"
visible_list="['left','right']"
maxdisp=192
num_train_epochs=70
max_train_steps=15000
gradient_accumulation_steps=1
checkpointing_steps=1000
lr_scheduler="cosine"


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --source_datapath $source_datapath \
          --outside_view_datapath $outside_view_datapath \
          --center_view_datapath $center_view_datapath \
          --confidence_datapath $confidence_datapath \
          --trainlist $trainlist \
          --vallist $vallist \
          --logging_dir $logging_dir \
          --tracker_project_name $tracker_project_name \
          --batch_size $batch_size \
          --datathread $datathread \
          --visible_list $visible_list \
          --maxdisp $maxdisp \
          --num_train_epochs $num_train_epochs \
          --max_train_steps $max_train_steps \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --checkpointing_steps $checkpointing_steps \
          --output_dir $output_dir \
          --lr_scheduler $lr_scheduler
          

}



TRAIN_KITTI_ORIGINAL

