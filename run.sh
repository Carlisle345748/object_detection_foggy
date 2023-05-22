python3 train_net.py \
      --num-gpus 1 \
      --config config/RCNN-C4-50-TS.yaml \
      OUTPUT_DIR output/cityscapes_ts_deb_high_weight

sleep 10m

runpodctl stop pod $RUNPOD_POD_ID
