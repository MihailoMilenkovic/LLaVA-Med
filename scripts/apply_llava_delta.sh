output_dir=$HOME/llava-weights-og
python3 -m llava.model.apply_delta \
    --base huggyllama/llama-7b \
    --target $output_dir \
    --delta microsoft/llava-med-7b-delta