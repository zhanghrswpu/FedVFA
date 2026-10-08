cd ..

current_path=$(pwd)
echo "当前工作路径为：$current_path"


batch_size=64
learning_rate=0.01
epochs=500
model="Net"
dataset="emnist"
weight_decay=0
local_epochs=1
client_count=100
print_acc_interval=10

enable_dirichlet=True

dirichlet_alpha=0.1
param="--enable_dirichlet $enable_dirichlet --client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

alpha=1
beta=0.05
variance=0.05


anchor_count=1
log_file_name="anchor_count=1"
python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count --log_file_name $log_file_name



#anchor_count=2
#log_file_name="anchor_count=2"
#python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count --log_file_name $log_file_name
#
#anchor_count=4
#log_file_name="anchor_count=4"
#python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count --log_file_name $log_file_name
#
##anchor_count=6
##log_file_name="anchor_count=6"
##python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count --log_file_name $log_file_name
#
#anchor_count=8
#log_file_name="anchor_count=8"
#python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count --log_file_name $log_file_name
#
#
#anchor_count=10
#log_file_name="anchor_count=10"
#python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count --log_file_name $log_file_name
#
#
#anchor_count=20
#log_file_name="anchor_count=20"
#python main.py $param --loss FedSubLoss --alpha $alpha --beta $beta --variance $variance --anchor_count $anchor_count --log_file_name $log_file_name
#
