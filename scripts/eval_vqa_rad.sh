script_dir=$(dirname "$(readlink -f "$0")")
par_dir=$(dirname "$script_dir")
weights_path_finetuned=/home/mmilenkovic/llava-weights-finetuned-vqa-rad 
weights_path_og=/home/mmilenkovic/llava-weights-og
use_finetuned_model=true

if [ "$use_finetuned_model" = "true" ]; then
    weights_path="$weights_path_finetuned"
else
    weights_path="$weights_path_og"
fi


answer_dir="/home/mmilenkovic/git/LLaVA-Med/data/finetune-vqa-rad-results.json"
test_question_folder="/home/mmilenkovic/git/LLaVA-Med/data/vqa_rad/test"
test_question_file="$test_question_folder/test.json"
test_image_path="$test_question_folder/images"
echo "loading weights from $weights_path"
echo "saving answers to $answer_dir"

python $par_dir/llava/eval/model_vqa.py \
    --model-name $weights_path\
    --question-file $test_question_file \
    --image-folder $test_image_path \
    --answers-file $answer_dir 
   