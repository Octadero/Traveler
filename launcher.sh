python3 traveller.py --gpu=0 --batch_size=75 --rnn_size=150 --checkpoint="./saved/model_no_gpu_75_150.ckpt" --logdir="/tmp/traveler/model_no_gpu_75_150/"
python3 traveller.py --gpu=0 --batch_size=150 --rnn_size=150 --checkpoint="./saved/model_no_gpu_150_150.ckpt" --logdir="/tmp/traveler/model_no_gpu_150_150/"

python3 traveller.py --gpu=0 --batch_size=75 --rnn_size=200 --checkpoint="./saved/model_no_gpu_75_200.ckpt" --logdir="/tmp/traveler/model_no_gpu_75_200/"
python3 traveller.py --gpu=0 --batch_size=150 --rnn_size=200 --checkpoint="./saved/model_no_gpu_150_200.ckpt" --logdir="/tmp/traveler/model_no_gpu_150_200/"

python3 traveller.py --gpu=0 --batch_size=75 --rnn_size=300 --checkpoint="./saved/model_no_gpu_75_300.ckpt" --logdir="/tmp/traveler/model_no_gpu_75_300/"
python3 traveller.py --gpu=0 --batch_size=150 --rnn_size=300 --checkpoint="./saved/model_no_gpu_150_300.ckpt" --logdir="/tmp/traveler/model_no_gpu_150_300/"

python3 traveller.py --gpu=0 --batch_size=75 --rnn_size=400 --checkpoint="./saved/model_no_gpu_75_300.ckpt" --logdir="/tmp/traveler/model_no_gpu_75_400/"
python3 traveller.py --gpu=0 --batch_size=150 --rnn_size=400 --checkpoint="./saved/model_no_gpu_150_300.ckpt" --logdir="/tmp/traveler/model_no_gpu_150_400/"

python3 traveller.py --gpu=1 --batch_size=75 --rnn_size=150 --checkpoint="./saved/model_gpu_75_150.ckpt" --logdir="/tmp/traveler/model_gpu_75_150/"
python3 traveller.py --gpu=1 --batch_size=150 --rnn_size=150 --checkpoint="./saved/model_gpu_150_150.ckpt" --logdir="/tmp/traveler/model_gpu_150_150/"

python3 traveller.py --gpu=1 --batch_size=75 --rnn_size=200 --checkpoint="./saved/model_no_gpu_75_200.ckpt" --logdir="/tmp/traveler/model_gpu_75_200/"
python3 traveller.py --gpu=1 --batch_size=150 --rnn_size=200 --checkpoint="./saved/model_no_gpu_150_200.ckpt" --logdir="/tmp/traveler/model_gpu_150_200/"

python3 traveller.py --gpu=1 --batch_size=75 --rnn_size=300 --checkpoint="./saved/model_gpu_75_110.ckpt" --logdir="/tmp/traveler/model_gpu_75_300/"
python3 traveller.py --gpu=1 --batch_size=150 --rnn_size=300 --checkpoint="./saved/model_gpu_150_110.ckpt" --logdir="/tmp/traveler/model_gpu_150_300/"

python3 traveller.py --gpu=1 --batch_size=75 --rnn_size=400 --checkpoint="./saved/model_gpu_75_110.ckpt" --logdir="/tmp/traveler/model_gpu_75_400/"
python3 traveller.py --gpu=1 --batch_size=150 --rnn_size=400 --checkpoint="./saved/model_gpu_150_110.ckpt" --logdir="/tmp/traveler/model_gpu_150_400/"

