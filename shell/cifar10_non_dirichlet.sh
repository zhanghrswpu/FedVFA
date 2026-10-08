cd ..

current_path=$(pwd)
echo "当前工作路径为：$current_path"


batch_size=64
learning_rate=0.01
epochs=1000
model="ResNet20"
dataset="cifar10"
weight_decay=0
local_epochs=1
print_acc_interval=10
enable_dirichlet=False
dirichlet_alpha=0.1

alpha=1
beta=0.05
anchor_count=6
variance=0.05

each_class_count=2
param="--each_class_count $each_class_count --enable_dirichlet $enable_dirichlet --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

python main.py $param --loss CE
python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count
python main.py $param --loss FedRs
python main.py $param --loss FedLC
python main.py $param --loss FedProx
python main.py $param --loss MoonLoss


each_class_count=5
param="--each_class_count $each_class_count --enable_dirichlet $enable_dirichlet --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"



# python main.py $param --loss CE
# python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count
# python main.py $param --loss FedRs
# python main.py $param --loss FedLC
# python main.py $param --loss FedProx
# python main.py $param --loss MoonLoss

