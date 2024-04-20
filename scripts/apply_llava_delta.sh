weights_path_og=$HOME/llava-weights-og
python3 -m llava.model.apply_delta \
    --base huggyllama/llama-7b \
    --target $weights_path_og \
    --delta microsoft/llava-med-7b-delta