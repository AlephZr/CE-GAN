import time
from itertools import cycle

import numpy as np
import tensorflow as tf
import torch

from algos.neuroevolution import SSNE
from core import utils
from core.tester import tester
from core.utils import generate_image, toy_generate_kde, toy_true_dist_kde, toy_generate_image, print_current_losses
from envs_repo.inception_pytorch.inception_utils import prepare_inception_metrics
from tflib import fid
from envs_repo.MMD_computer import compute_metric_mmd2, MMD_loss


class CEGAN_Trainer:

    def __init__(self, args, model_constructor, env_constructor):
        self.args = args
        self.device = args.device
        self.env = env_constructor
        self.D_train_sample = args.batch_size * args.D_iters
        self.batch_sample1 = self.D_train_sample + args.batch_size if 'rsgan' in args.g_loss_mode else self.D_train_sample
        self.batch_sample2 = self.batch_sample1 + args.eval_size if 'fitness' in args.g_loss_mode else self.batch_sample1

        # PG Learner
        from algos.gan import GAN
        self.learner = GAN(args, model_constructor, env_constructor)

        # Evolution
        self.evolver = SSNE(self.args, self.learner.netG, self.learner.optimizerG, self.learner.netD)

        self.no_FID = True
        self.no_IS = True
        self.sess, self.mu_real, self.sigma_real, self.get_inception_metrics = None, None, None, None
        if self.args.use_pytorch_scores and self.args.test_name:
            parallel = False
            if 'FID' in args.test_name:
                self.no_FID = False
            if 'IS' in args.test_name:
                self.no_IS = False
            self.get_inception_metrics = prepare_inception_metrics(args.dataset_name, parallel, self.no_IS, self.no_FID)
        else:
            if 'FID' in args.test_name:
                if self.args.dataset_name == 'CIFAR10':
                    STAT_FILE = './tflib/TTUR/stats/fid_stats_cifar10_train.npz'
                elif self.args.dataset_name == 'CelebA':
                    STAT_FILE = './tflib/TTUR/stats/fid_stats_celeba.npz'
                elif self.args.dataset_name == 'LSUN':
                    STAT_FILE = './tflib/TTUR/stats/fid_stats_lsun_train.npz'
                INCEPTION_PATH = './tflib/IS/imagenet'

                print("load train stats.. ")
                # load precalculated training set statistics
                f = np.load(STAT_FILE)
                self.mu_real, self.sigma_real = f['mu'][:], f['sigma'][:]
                f.close()
                print("ok")

                inception_path = fid.check_or_download_inception(INCEPTION_PATH)  # download inception network
                fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                self.sess.run(tf.global_variables_initializer())

                self.no_FID = False
            if 'IS' in args.test_name:
                self.no_IS = False

    def forward_generation(self, gen, tracker, D_train_images, rsgan_train_images, fitness_train_images, eval_images):

        # # Sets the number of threads used for intraop parallelism on CPU.
        # torch.set_num_threads(4)

        # NeuroEvolution's probabilistic selection and recombination step
        gen_samples, selected_operator = self.evolver.epoch(gen, rsgan_train_images, fitness_train_images, eval_images)

        ############# UPDATE PARAMS USING GRADIENT DESCENT ##########
        self.learner.update_D_parameters(gen_samples, D_train_images)

        # # Compute the champion's eplen
        # champ_len = all_eplens[all_fitness.index(max(all_fitness))] if self.args.pop_size > 1 else rollout_eplens[
        #     rollout_fitness.index(max(rollout_fitness))]

        return selected_operator

    def train(self, iterations_limit):
        # Define Tracker class to track scores
        test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag],
                                     '.csv')  # Tracker class to log progress

        if self.args.dataset_name == '8gaussians' or self.args.dataset_name == '25gaussians' or self.args.dataset_name == 'swissroll':
            mmd_computer = MMD_loss()

        time_start = time.time()

        epoch_num = 0
        gen = 1
        while gen < iterations_limit + 1:  # Infinite generations
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration

            for _, real_samples in enumerate(self.env.train_dataset):
                iter_start_time = time.time()  # timer for computation per iteration

                if self.args.dataset_name != '8gaussians' and self.args.dataset_name != '25gaussians' and self.args.dataset_name != 'swissroll':
                    real_samples = real_samples[0]
                else:
                    test_samples = real_samples[:512]
                    real_samples = real_samples[512:]

                # Train one iteration
                selected_operator = self.forward_generation(gen, test_tracker,
                                                            real_samples[:self.D_train_sample].to(
                                                                device=self.device),
                                                            real_samples[self.D_train_sample:self.batch_sample1].to(
                                                                device=self.device),
                                                            real_samples[self.batch_sample1:self.batch_sample2].to(
                                                                device=self.device),
                                                            real_samples[-self.args.eval_size:].to(
                                                                device=self.device))

                if gen % 1000 == 0:
                    t_data = iter_start_time - iter_data_time
                    t_comp = (time.time() - iter_start_time) / self.args.batch_size
                    utils.print_current_losses(epoch_num, gen, t_comp, t_data, selected_operator)

                    # print('Gen:', gen, 'selected_operator:', selected_operator, ' GPS:',
                    #       '%.2f' % (gen / (time.time() - time_start)), ' IS_score u/std',
                    #       utils.pprint(IS_mean) if IS_mean is not None else None,
                    #       utils.pprint(IS_var) if IS_var is not None else None,
                    #       ' FID_score', utils.pprint(FID) if FID is not None else None)
                    # print()
                # if gen % 100 == 0:
                #     self.args.writer.add_scalar('GPS', gen / (time.time() - time_start), gen)

                ###### TEST SCORE ######
                if gen % 5000 == 0:
                    self.learner.netG.load_state_dict(self.evolver.genes[-1])
                    torch.save(self.learner.netG,
                               './checkpoint/{0}/netG_{1}.pth'.format(self.args.dataset_name, gen))

                if self.args.test_name and gen % self.args.test_frequency == 0:
                    # FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST
                    self.learner.netG.load_state_dict(self.evolver.genes[-1])

                    scores = tester(self.args, self.learner.netG, not self.no_FID, not self.no_IS, self.sess,
                                    self.mu_real, self.sigma_real, self.get_inception_metrics)

                    if not self.no_IS:
                        test_tracker.update([scores['IS_mean']], gen)
                        test_tracker.update([scores['IS_var']], gen)
                        self.args.writer.add_scalar('IS_score', scores['IS_mean'], gen)
                    if not self.no_FID:
                        test_tracker.update([scores['FID']], gen)
                        self.args.writer.add_scalar('FID_score', scores['FID'], gen)
                    utils.print_current_scores(epoch_num, gen, scores)

                if gen % 1000 == 0:
                    self.learner.netG.load_state_dict(self.evolver.genes[-1])
                    if self.args.dataset_name == '8gaussians' or self.args.dataset_name == '25gaussians' or self.args.dataset_name == 'swissroll':
                        # toy_true_dist_kde(self.args, real_samples)
                        utils.toy_generate_kde(self.args, self.learner.netG)

                        with torch.no_grad():
                            noisev = torch.randn(512, self.args.z_dim, device=self.args.device)
                        gen_samples = self.learner.netG(noisev).detach()
                        test_samples = test_samples.to(device=self.device)
                        mmd2 = mmd_computer.forward(gen_samples, test_samples)
                        # mmd2 = abs(compute_metric_mmd2(gen_samples, test_samples))
                        test_tracker.update([mmd2], gen)
                        self.args.writer.add_scalar('mmd2', mmd2, gen)

                        test_samples = test_samples.detach().cpu().numpy()
                        gen_samples = gen_samples.detach().cpu().numpy()
                        utils.toy_generate_image(self.args, test_samples, gen_samples)
                    else:
                        utils.generate_image(gen, self.args, self.learner.netG)

                gen += 1

            epoch_num += 1
            print('(epoch_%d) End of giters %d / %d \t Time Taken: %d sec' % (
                epoch_num, gen, iterations_limit, time.time() - epoch_start_time))

        self.args.writer.close()
