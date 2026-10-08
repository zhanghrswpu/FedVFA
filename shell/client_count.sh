cd ..

current_path=$(pwd)
echo "当前工作路径为：$current_path"

batch_size=64
learning_rate=0.01
epochs=400
model="LeNet"
dataset="mnist"
weight_decay=0
local_epochs=1
client_count=100
print_acc_interval=10

enable_dirichlet=True

dirichlet_alpha=0.1

classifier_iter=0
beta=5
enable_after_adjust=True
fedvfa_alpha=0.01

client_count=50
param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

log_file_name="FedAvg_FedFA_client_$client_count.json"
python main.py $param --loss FedFA --log_file_name $log_file_name
log_file_name="FedAvg_CE_client_$client_count.json"
python main.py $param --loss CE --log_file_name $log_file_name
log_file_name="FedAvg_FedRs_client_$client_count.json"
python main.py $param --loss FedRs --log_file_name $log_file_name
log_file_name="FedAvg_FedLC_client_$client_count.json"
python main.py $param --loss FedLC --log_file_name $log_file_name
log_file_name="FedAvg_FedProx_client_$client_count.json"
python main.py $param --loss FedProx --log_file_name $log_file_name
log_file_name="FedAvg_MoonLoss_client_$client_count.json"
python main.py $param --loss MoonLoss --log_file_name $log_file_name
log_file_name="FedAvg_FedVFA_client_$client_count.json"
python main.py $param --loss FedVFA --alpha $fedvfa_alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

client_count=150
param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

log_file_name="FedAvg_FedFA_client_$client_count.json"
python main.py $param --loss FedFA --log_file_name $log_file_name
log_file_name="FedAvg_CE_client_$client_count.json"
python main.py $param --loss CE --log_file_name $log_file_name
log_file_name="FedAvg_FedRs_client_$client_count.json"
python main.py $param --loss FedRs --log_file_name $log_file_name
log_file_name="FedAvg_FedLC_client_$client_count.json"
python main.py $param --loss FedLC --log_file_name $log_file_name
log_file_name="FedAvg_FedProx_client_$client_count.json"
python main.py $param --loss FedProx --log_file_name $log_file_name
log_file_name="FedAvg_MoonLoss_client_$client_count.json"
python main.py $param --loss MoonLoss --log_file_name $log_file_name
log_file_name="FedAvg_FedVFA_client_$client_count.json"
python main.py $param --loss FedVFA --alpha $fedvfa_alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

client_count=200
param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

log_file_name="FedAvg_FedFA_client_$client_count.json"
python main.py $param --loss FedFA --log_file_name $log_file_name
log_file_name="FedAvg_CE_client_$client_count.json"
python main.py $param --loss CE --log_file_name $log_file_name
log_file_name="FedAvg_FedRs_client_$client_count.json"
python main.py $param --loss FedRs --log_file_name $log_file_name
log_file_name="FedAvg_FedLC_client_$client_count.json"
python main.py $param --loss FedLC --log_file_name $log_file_name
log_file_name="FedAvg_FedProx_client_$client_count.json"
python main.py $param --loss FedProx --log_file_name $log_file_name
log_file_name="FedAvg_MoonLoss_client_$client_count.json"
python main.py $param --loss MoonLoss --log_file_name $log_file_name
log_file_name="FedAvg_FedVFA_client_$client_count.json"
python main.py $param --loss FedVFA --alpha $fedvfa_alpha --beta $beta --classifier_iter $classifier_iter --enable_after_adjust $enable_after_adjust --log_file_name $log_file_name

# dirichlet_alpha=0.2
# client_count=50
# log_file_name="FedAvg_FedFA_client_$client_count.json"
# param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

# log_file_name="FedAvg_FedFA_client_$client_count.json"
# python main.py $param --loss FedFA --log_file_name $log_file_name
# log_file_name="FedAvg_CE_client_$client_count.json"
# python main.py $param --loss CE --log_file_name $log_file_name
# log_file_name="FedAvg_FedRs_client_$client_count.json"
# python main.py $param --loss FedRs --log_file_name $log_file_name
# log_file_name="FedAvg_FedLC_client_$client_count.json"
# python main.py $param --loss FedLC --log_file_name $log_file_name
# log_file_name="FedAvg_FedProx_client_$client_count.json"
# python main.py $param --loss FedProx --log_file_name $log_file_name
# log_file_name="FedAvg_MoonLoss_client_$client_count.json"
# python main.py $param --loss MoonLoss --log_file_name $log_file_name
# log_file_name="FedAvg_FedVFA_client_$client_count.json"
# python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter  --log_file_name $log_file_name

# client_count=150
# param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

# log_file_name="FedAvg_FedFA_client_$client_count.json"
# python main.py $param --loss FedFA --log_file_name $log_file_name
# log_file_name="FedAvg_CE_client_$client_count.json"
# python main.py $param --loss CE --log_file_name $log_file_name
# log_file_name="FedAvg_FedRs_client_$client_count.json"
# python main.py $param --loss FedRs --log_file_name $log_file_name
# log_file_name="FedAvg_FedLC_client_$client_count.json"
# python main.py $param --loss FedLC --log_file_name $log_file_name
# log_file_name="FedAvg_FedProx_client_$client_count.json"
# python main.py $param --loss FedProx --log_file_name $log_file_name
# log_file_name="FedAvg_MoonLoss_client_$client_count.json"
# python main.py $param --loss MoonLoss --log_file_name $log_file_name
# log_file_name="FedAvg_FedVFA_client_$client_count.json"
# python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter  --log_file_name $log_file_name

# client_count=200
# param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

# log_file_name="FedAvg_FedFA_client_$client_count.json"
# python main.py $param --loss FedFA --log_file_name $log_file_name
# log_file_name="FedAvg_CE_client_$client_count.json"
# python main.py $param --loss CE --log_file_name $log_file_name
# log_file_name="FedAvg_FedRs_client_$client_count.json"
# python main.py $param --loss FedRs --log_file_name $log_file_name
# log_file_name="FedAvg_FedLC_client_$client_count.json"
# python main.py $param --loss FedLC --log_file_name $log_file_name
# log_file_name="FedAvg_FedProx_client_$client_count.json"
# python main.py $param --loss FedProx --log_file_name $log_file_name
# log_file_name="FedAvg_MoonLoss_client_$client_count.json"
# python main.py $param --loss MoonLoss --log_file_name $log_file_name
# log_file_name="FedAvg_FedVFA_client_$client_count.json"
# python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter  --log_file_name $log_file_name

# dirichlet_alpha=0.5
# client_count=50
# param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

# log_file_name="FedAvg_FedFA_client_$client_count.json"
# python main.py $param --loss FedFA --log_file_name $log_file_name
# log_file_name="FedAvg_CE_client_$client_count.json"
# python main.py $param --loss CE --log_file_name $log_file_name
# log_file_name="FedAvg_FedRs_client_$client_count.json"
# python main.py $param --loss FedRs --log_file_name $log_file_name
# log_file_name="FedAvg_FedLC_client_$client_count.json"
# python main.py $param --loss FedLC --log_file_name $log_file_name
# log_file_name="FedAvg_FedProx_client_$client_count.json"
# python main.py $param --loss FedProx --log_file_name $log_file_name
# log_file_name="FedAvg_MoonLoss_client_$client_count.json"
# python main.py $param --loss MoonLoss --log_file_name $log_file_name
# log_file_name="FedAvg_FedVFA_client_$client_count.json"
# python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter  --log_file_name $log_file_name

# client_count=150
# param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

# log_file_name="FedAvg_FedFA_client_$client_count.json"
# python main.py $param --loss FedFA --log_file_name $log_file_name
# log_file_name="FedAvg_CE_client_$client_count.json"
# python main.py $param --loss CE --log_file_name $log_file_name
# log_file_name="FedAvg_FedRs_client_$client_count.json"
# python main.py $param --loss FedRs --log_file_name $log_file_name
# log_file_name="FedAvg_FedLC_client_$client_count.json"
# python main.py $param --loss FedLC --log_file_name $log_file_name
# log_file_name="FedAvg_FedProx_client_$client_count.json"
# python main.py $param --loss FedProx --log_file_name $log_file_name
# log_file_name="FedAvg_MoonLoss_client_$client_count.json"
# python main.py $param --loss MoonLoss --log_file_name $log_file_name
# log_file_name="FedAvg_FedVFA_client_$client_count.json"
# python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter  --log_file_name $log_file_name

# client_count=200
# param="--client_count $client_count --dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

# log_file_name="FedAvg_FedFA_client_$client_count.json"
# python main.py $param --loss FedFA --log_file_name $log_file_name
# log_file_name="FedAvg_CE_client_$client_count.json"
# python main.py $param --loss CE --log_file_name $log_file_name
# log_file_name="FedAvg_FedRs_client_$client_count.json"
# python main.py $param --loss FedRs --log_file_name $log_file_name
# log_file_name="FedAvg_FedLC_client_$client_count.json"
# python main.py $param --loss FedLC --log_file_name $log_file_name
# log_file_name="FedAvg_FedProx_client_$client_count.json"
# python main.py $param --loss FedProx --log_file_name $log_file_name
# log_file_name="FedAvg_MoonLoss_client_$client_count.json"
# python main.py $param --loss MoonLoss --log_file_name $log_file_name
# log_file_name="FedAvg_FedVFA_client_$client_count.json"
# python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter  --log_file_name $log_file_name