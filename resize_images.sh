N=30
(
for j in data/bdd/bdd100k_dataset_321x185/*/*/*.png; do
   ((i=i%N)); ((i++==0)) && wait
 mogrify -interpolate nearest-neighbor -resize 321x185 $j &
done
)