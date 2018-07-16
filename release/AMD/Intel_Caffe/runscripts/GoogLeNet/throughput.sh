for filename in $PWD/*out_*; do
        NAME=$(basename "$filename")
        echo $NAME
        sh script.sh $NAME
done

