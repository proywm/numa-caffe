grep -Ri -C 0 " Iteration 800, loss =" $* >> temp
grep -Ri -C 0 "Iteration 0, lr =" $* >> temp
awk -F ' ' '{print $2}' temp > temp1
rm temp
sed -e "s/:/ /g" < temp1 > temp2
cat temp2
rm temp1
awk '{if(NR==1){ end=($1*60*60)+($2*60)+$3;};begin=($1*60*60)+($2*60)+$3;}END{diff=end-begin; print diff; printf "Throughput %0.2f (measured with 800 iterations)\n",800/diff}' <temp2

rm temp2
