
# for dataset in YeastH.reorder OVCAR-8H.reorder Yeast.reorder DD.reorder web-BerkStan.reorder reddit.reorder ddi.reorder protein.reorder
for dataset in rma10
do
  python -u run_DTC_SpMM.py --dataset ${dataset}
done
