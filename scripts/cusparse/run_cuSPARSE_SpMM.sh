export CUDA_VISIBLE_DEVICES="1"
# for dataset in YeastH.reorder OVCAR-8H.reorder Yeast.reorder DD.reorder web-BerkStan.reorder reddit.reorder ddi.reorder protein.reorder
for dataset in YeastH.reorder cop20k_A
do
  python -u run_cuSPARSE.py --dataset ${dataset}
done
