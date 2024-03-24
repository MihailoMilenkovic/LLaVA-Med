script_dir=$(dirname "$(readlink -f "$0")")
par_dir=$(dirname "$script_dir")

python $script_dir/convert_vqa_rad_to_llava.py \
  --train_dir=$par_dir/data/vqa_rad/train \
  --test_dir=$par_dir/data/vqa_rad/test
