if [ $# -lt 1 ]
then
    echo 'Usage: urlFile'
    exit
fi

cnt=0
eval 'a=(`cat $1`)'
echo Total: ${#a[*]} urls
for x in ${a[*]}
do
    wget --tries 3 -T 10 -q $x
    if [[ $? == 0 ]]
    then
	((cnt++))
    fi
done

echo Downloaded: $cnt pictures
