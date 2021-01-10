set -ex
python test.py --dataset_name CIFAR10 \
       --gpu_id 1 \
       --netG EGAN32 --ngf 128 \
       --z_dim 100 \
       --test_name IS --test_size 50000 --fid_batch_size 100 \
       --savetag nsgan_lr=1e-4 \
