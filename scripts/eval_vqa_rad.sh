script_dir=$(dirname "$(readlink -f "$0")")
par_dir=$(dirname "$script_dir")
weights_path_finetuned=$HOME/llava-weights-finetuned-vqa-rad 
weights_path_og=$HOME/llava-weights-og
repo_dir=$HOME/git/LLaVA-Med

use_finetuned_model=true

if [ "$use_finetuned_model" = "true" ]; then
    weights_path="$weights_path_finetuned"
    model_type="base"
else
    weights_path="$weights_path_og"
    model_type="finetuned"
fi


answer_path="$repo_dir/data/finetune-vqa-rad-results-$model_type.json"
test_question_folder="$repo_dir/data/vqa_rad/test"
test_question_file="$test_question_folder/questions.jsonl"
test_image_path="$test_question_folder/images"
echo "loading weights from $weights_path"
echo "saving answers to $answer_path"

generate_new_answers=true

if [ "$generate_new_answers" = "true" ]; then
    python $par_dir/llava/eval/model_vqa.py \
        --model-name $weights_path\
        --question-file $test_question_file \
        --image-folder $test_image_path \
        --answers-file $answer_path 
fi

python $script_dir/process_vqa_rad_answers.py \
    --answers_file $answer_path \
    --test_data_file $test_question_folder/data.json

   