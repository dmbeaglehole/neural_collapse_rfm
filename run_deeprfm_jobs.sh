# datasets=("cifar" "svhn" "mnist")
datasets=("mnist")
n=50000
depth=20
sigmas=(0.05 2.0 0.1)

for dataset in ${datasets[@]};
do
    for sigma in ${sigmas[@]};
    do
        sbatch  --job-name="deep_rfm_nc_$dataset" --gpus=1 delta_setup "python -u deep_rfm.py -dataset $dataset -n $n -depth $depth -use_rff -width 4096 -sigma $sigma"
    done
done


# python -u deep_rfm.py -dataset mnist -n 50000 -depth 20 -use_rff -width 4096 -sigma 0.05
