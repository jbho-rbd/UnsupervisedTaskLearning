for i in {0..1}
do
sed 's/;//g' run$i > run$i.dat
rm run$i
done
