
# define array for number of epochs to search
epochs=(3 9)
# define array for number of epoch for learning rate
learning_rates=(3e-6 1e-5 2e-5 5e-5 1e-4)

# loop through the number of epochs
for epoch in "${epochs[@]}"
do
    for lr in "${learning_rates[@]}"
    do
        /bin/bash ./train_vqa_rad.sh --epochs $epoch --lr $lr
    done
done
    
