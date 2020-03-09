N=30
(
for j in data/bdd/bdd100k_dataset_321x185/*/images/*.jpg; do
   ((i=i%N)); ((i++==0)) && wait
 mogrify -format png $j &
done
)