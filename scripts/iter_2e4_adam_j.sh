TRAIN_KITTI(){
pretrain_name=PAMSDFRender
cd ..
mkdir logs
loss=config/loss_config_disp.json
outf_model=models_saved/$pretrain_name
logf=logs/$pretrain_name
# datapath=/data/public_dataset/KITTI_Raw_Data
datapath=/hdd/zsy_data/kitti_data
datathread=4
lr=2e-4
devices=0
dataset=KITTI
trainlist=filenames/kitti_raw_unos.txt
# tetainlist=filenames/KITTI_2015_train.txt    
vallist=filenames/KITTI_2015_train.txt
startR=0
startE=0
batchSize=8
testbatch=2
maxdisp=-1
save_logdir=experiments_logdir/$pretrain_name
model=${pretrain_name}
pretrain=none
initial_pretrain=none
sdf_type=MLP
summary_freq=10
sdf_weight=0.01

CUDA_VISIBLE_DEVICES=1 python3 -W ignore train_iter.py --cuda --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain \
               --initial_pretrain $initial_pretrain \
               --sdf_type $sdf_type \
               --summary_freq $summary_freq \
               --sdf_weight $sdf_weight \
	       --optimizer Adam \
	       --load_wandb
}

TRAIN_KITTI


