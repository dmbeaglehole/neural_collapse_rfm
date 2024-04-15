datasets=("svhn" "mnist" "cifar")

# lrs=(5e-2)
# n=50000
# epochs=500
# inits=(0.25)
# depth=5
# model="resnet18"
# opt="sgd"

lrs=(5e-5)
n=50000
epochs=250
inits=(0.1)
depth=5
model="mlp"
opt="adam"

for init in ${inits[@]};
do
    for lr in ${lrs[@]};
    do
        for dataset in ${datasets[@]};
        do
            sbatch  --job-name="nc_$dataset""_lr_$lr" --gpus=1 delta_setup "python -u nc_nn.py -dataset $dataset -n $n -lr $lr -epochs $epochs -init $init -opt $opt -depth $depth -measure_every 10 -model $model"
        done
    done
done



# python -u nc_nn.py -dataset cifar -n 50000 -lr 1e-4 -epochs 100 -opt adam -depth 5 -measure_every 10 -model mlp -init 0.1