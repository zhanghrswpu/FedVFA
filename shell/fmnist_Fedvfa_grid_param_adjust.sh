cd ..

current_path=$(pwd)
echo "当前工作路径为：$current_path"

batch_size=64
learning_rate=0.01
epochs=800
model="Net"
dataset="fmnist"
weight_decay=0
local_epochs=1
client_count=100
print_acc_interval=10
enable_dirichlet=True
dirichlet_alpha=0.1
param="--enable_dirichlet $enable_dirichlet --client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"
classifier_iter=0
enable_after_adjust=True

#beta=0.1
#
#alpha=0.001
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.005
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.01
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.05
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.1
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#beta=0.5
#
#alpha=0.001
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.005
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.01
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.05
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.1
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#beta=1
#
#alpha=0.001
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.005
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.01
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.05
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.1
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#beta=5
#
#alpha=0.001
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.005
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.01
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.05
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
#
#alpha=0.1
#log_file_name="beta=$beta,alpha=$alpha"
#python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name


beta=0

alpha=0.001
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

alpha=0.005
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

alpha=0.01
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

alpha=0.05
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

alpha=0.1
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name


beta=0.1
alpha=0
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

beta=0.5
alpha=0
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

beta=1
alpha=0
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

beta=5
alpha=0
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name


beta=0
alpha=0
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name
