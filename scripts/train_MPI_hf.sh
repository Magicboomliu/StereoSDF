TRAIN_MPI_ORIGINAL(){
cd ..
cd HuggingFace_Trainer/
datapath="/media/zliu/data12/dataset/Sintel/"
trainlist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/stereonet_logs"
output_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/simple_stereonet"
tracker_project_name="tracker_project_MPI"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe']"
maxdisp=192
num_train_epochs=70
max_train_steps=20000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='simple'
network_type="StereoNet"


CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
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
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate \
          --network_type $network_type
          

}


TRAIN_MPI_Plus_Outside(){
cd ..
cd HuggingFace_Trainer/
datapath="/media/zliu/data12/dataset/Sintel/"
trainlist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/logs_plus"
output_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/plus_outside_stereonet"
tracker_project_name="tracker_project_MPI_Outside"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_left_left','rendered_right_right']"
maxdisp=192
num_train_epochs=70
max_train_steps=20000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_outside'
network_type="StereoNet"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
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
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate \
          --network_type $network_type
          

}


TRAIN_MPI_Plus_Center(){
cd ..
cd HuggingFace_Trainer/
datapath="/media/zliu/data12/dataset/Sintel/"
trainlist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/logs_center"
output_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/plus_center_stereonet"
tracker_project_name="tracker_project_MPI_Center"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_med']"
maxdisp=192
num_train_epochs=70
max_train_steps=20000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_center'
network_type="StereoNet"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
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
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate \
          --network_type $network_type
          

}



TRAIN_MPI_Plus_OutSide_Center(){
cd ..
cd HuggingFace_Trainer/
datapath="/media/zliu/data12/dataset/Sintel/"
trainlist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt"
vallist="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt"
logging_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/logs_center_outside"
output_dir="/home/zliu/Desktop/NIPS2024/UnsupervisedStereo/StereoSDF/outputs/plus_outside_center_stereonet"
tracker_project_name="tracker_project_MPI_OutSide_Center"
batch_size=4
datathread=4
visible_list="['final_left','final_right','disp','occlusions','outofframe','rendered_med','rendered_left_left','rendered_right_right']"
maxdisp=192
num_train_epochs=70
max_train_steps=20000
gradient_accumulation_steps=1
checkpointing_steps=1000
learning_rate=1e-4
lr_scheduler="cosine"
loss_type='plus_outside_center'
network_type="StereoNet"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="no"   unsupervised_stereo_matching_trainer.py \
          --datapath $datapath \
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
          --lr_scheduler $lr_scheduler \
          --loss_type $loss_type \
          --learning_rate $learning_rate \
          --network_type $network_type
          

}







TRAIN_MPI_Plus_OutSide_Center

# TRAIN_MPI_Plus_Center

# TRAIN_MPI_ORIGINAL
# TRAIN_MPI_Plus_Center
# TRAIN_MPI_Plus_Outside