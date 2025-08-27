python /data2/peilincai/Difix3D/src/build_dataset_json.py \
  --ghost_dir /data2/peilincai/Difix3D/datasets/ghosts \
  --label_dir /data2/peilincai/Difix3D/datasets/label \
  --output_json /data2/peilincai/Difix3D/datasets/difix3d.json \
  --prompt "remove degradation" \
  --test_ratio 0.1 \
  --seed 42 \
  --both_refs