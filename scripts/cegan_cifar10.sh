set -ex
python train.py --eval_criteria discriminator \
       --dataset_name CIFAR10 \
       --batchsize 32 --evalsize 256 \
       --netD EGAN32 --netG EGAN32 --ngf 128 --ndf 128 \
       --discriminator_lr 0.0001 --generator_lr 0.0001 \
       --z_dim 100 \
       --crop_size 32 --load_size 32 \
       --popsize 1 --crosssize 1 \
       --d_loss_mode vanilla --g_loss_mode nsgan lsgan vanilla \
       --D_iters 3 \
       --test_name IS --test_size 50000 --fid_batch_size 500 --test_frequency 5000 \
       --use_comprehensive_selection --use_gp \
       --use_pytorch_scores \
       --total_iterations 100 \
       --savetag CEGAN_gp \
