
# Record relevant topics
if [[ -z $1 ]];
then    
    echo "no param passed"
else
    #echo $2
    rosbag record $@
fi

