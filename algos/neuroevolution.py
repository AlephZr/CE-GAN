import copy
import numpy as np
import torch
from scipy.special import comb

from core.runner import Evaluator
from core.utils import dict_MinMaxScaler


class SSNE:

    def __init__(self, args, generator, optimizer, discriminator):
        self.args = args
        self.device = args.device
        self.input_dim = args.z_dim

        self.g_loss_number = len(args.g_loss_mode)
        self.mutate_size = args.pop_size * self.g_loss_number
        max_crossover = int(comb(self.mutate_size, 2))
        self.crossover_size = args.crossover_size if max_crossover >= args.crossover_size else max_crossover
        print('d_loss_mode:', args.d_loss_mode, 'g_loss_mode:', args.g_loss_mode, 'mutate_size:', self.mutate_size,
              'crossover_size:', self.crossover_size)

        self.individual = generator
        self.individual_optimizer = optimizer
        self.critic = discriminator
        self.genes = []
        self.genes_optimizer = []
        for _ in range(args.pop_size):
            self.genes.append(copy.deepcopy(self.individual.state_dict()))
            self.genes_optimizer.append(copy.deepcopy(self.individual_optimizer.state_dict()))
        self.env = Evaluator(args, generator, discriminator)

        # self.mutate_evaluate = False if len(list(combinations(range(len(self.mutate_pop)), 2))) <= len(
        #     self.crossover_pop) else True

        self.writer = args.writer
        # GAN TRACKERS
        self.selection_stats = {'total': 0, 'crossover': 0, 'parents': 0}
        for g_loss in args.g_loss_mode:
            self.selection_stats[g_loss] = 0

        # Adversarial ground truths
        self.ones_label = torch.ones(args.batch_size, device=self.device)
        self.zeros_label = torch.zeros(args.batch_size, device=self.device)
        # Loss function
        self.BCEL_loss = torch.nn.BCEWithLogitsLoss()
        self.MSE_loss = torch.nn.MSELoss()

    def distilation_crossover(self, noise, gene1, gene2, gene1_optim, gene2_optim, gene1_result, gene2_result,
                              offspring, offspring_optim):
        for p in self.critic.parameters():
            p.requires_grad = False  # to avoid computation

        self.individual.load_state_dict(gene1)
        self.individual_optimizer.load_state_dict(gene1_optim)

        self.individual_optimizer.zero_grad()

        gene1_fake = gene1_result[1]
        gene2_fake = gene2_result[1]
        gene1_q = gene1_result[0].flatten()
        gene2_q = gene2_result[0].flatten()

        eps = 0.0
        fake_batch = torch.cat((gene1_fake[gene1_q - gene2_q > eps], gene2_fake[gene2_q - gene1_q >= eps])).detach()
        noise_batch = torch.cat((noise[gene1_q - gene2_q > eps], noise[gene2_q - gene1_q >= eps]))
        offspring_fake = self.individual(noise_batch)

        # Actor Update
        # offspring_optim.zero_grad()
        # sq = (offspring_fake - fake_batch) ** 2
        # policy_loss = torch.sum(sq) + torch.mean(offspring_fake ** 2)
        policy_loss = self.MSE_loss(offspring_fake, fake_batch)
        policy_loss.backward()
        self.individual_optimizer.step()

        offspring.append(copy.deepcopy(self.individual.state_dict()))
        offspring_optim.append(copy.deepcopy(self.individual_optimizer.state_dict()))

    def gradient_mutate(self, mutate_pop, mutate_optim, real_samples1, real_samples2, gene, optimizer, mode):
        batch_size = self.args.batch_size
        for p in self.critic.parameters():
            p.requires_grad = False  # to avoid computation

        if self.args.netG == 'DCGAN':
            noise = torch.randn(batch_size, self.input_dim, 1, 1, device=self.device)
        elif self.args.netG == 'WGAN':
            noise = torch.randn(batch_size, self.input_dim, device=self.device)
        elif self.args.netG == 'FC2':
            noise = torch.randn(batch_size, self.input_dim, device=self.device)
        elif 'EGAN' in self.args.netG:
            noise = torch.rand(batch_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
        else:
            raise NotImplementedError('netG [%s] is not found' % self.args.netG)

        # Variation
        for g_loss in mode:
            self.individual.load_state_dict(gene)
            self.individual_optimizer.load_state_dict(optimizer)
            self.individual_optimizer.zero_grad()

            if g_loss == 'fitness':  # Fitness function Variation
                # noise = torch.rand(self.args.eval_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
                noise = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
                gen_samples = self.individual(noise)
                gan_fitness = self.env.get_fitness(gen_samples, real_samples2)
                gan_loss = -gan_fitness
            else:
                gen_samples = self.individual(noise)
                gen_critic = self.critic(gen_samples)
                if g_loss == 'nsgan':  # nsgan Variation
                    gan_loss = self.BCEL_loss(gen_critic, self.ones_label)
                elif g_loss == 'vanilla':  # vanilla Variation
                    gan_loss = -self.BCEL_loss(gen_critic, self.zeros_label)
                elif g_loss == 'lsgan':  # lsgan Variation
                    gan_loss = self.MSE_loss(gen_critic, self.ones_label)
                elif g_loss == 'wgan':  # wgan Variation
                    gan_loss = -gen_critic.mean()
                elif g_loss == 'rsgan':  # rsgan Variation
                    real_critic = self.critic(real_samples1)
                    gan_loss = self.BCEL_loss(gen_critic - real_critic, self.ones_label)
                else:
                    raise NotImplementedError('gan mode %s not implemented' % g_loss)
            gan_loss.backward()
            self.individual_optimizer.step()

            mutate_pop.append(copy.deepcopy(self.individual.state_dict()))
            mutate_optim.append(copy.deepcopy(self.individual_optimizer.state_dict()))

    @staticmethod
    def sort_groups_by_fitness(genomes, fitness):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i + 1:]:
                if fitness[first] < fitness[second]:
                    groups.append((second, first, fitness[first] + fitness[second]))
                else:
                    groups.append((first, second, fitness[first] + fitness[second]))
        return sorted(groups, key=lambda group: group[2], reverse=True)

    def epoch(self, gen, rsgan_train_images, fitness_train_images, eval_images):
        mode = self.args.g_loss_mode
        mode_number = self.g_loss_number

        if self.args.netG == 'DCGAN':
            noise_mutate = torch.randn(self.args.eval_size, self.input_dim, 1, 1, device=self.device)
            noise_crossover = torch.randn(self.args.eval_size, self.input_dim, 1, 1, device=self.device)
        elif 'EGAN' in self.args.netG:
            noise_mutate = torch.rand(self.args.eval_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
            noise_crossover = torch.rand(self.args.eval_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
        elif self.args.netG == 'WGAN':
            noise_mutate = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
            noise_crossover = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
        elif self.args.netG == 'FC2':
            noise_mutate = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
            noise_crossover = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
        else:
            raise NotImplementedError('netG [%s] is not found' % self.args.netG)

        ########## PARENTS ############
        if self.args.eval_parents:
            # Initialize mutate population
            parents_pop = copy.deepcopy(self.genes)
            parents_optim = copy.deepcopy(self.genes_optimizer)

            # Evaluation
            parents_fitness_evals = []
            parents_critics_samples = []
            for i in range(self.args.pop_size):
                self.individual.load_state_dict(parents_pop[i])
                parents_fitness, parents_gen_critic, parents_gen_images = self.env.eval_worker('parents', noise_mutate,
                                                                                               eval_images)
                parents_fitness_evals.append(parents_fitness)
                parents_critics_samples.append((parents_gen_critic, parents_gen_images))

        ########## MUTATION ############
        # Initialize mutate population
        mutate_pop = []
        mutate_optim = []

        # Mutate all genes in the population
        for i in range(self.args.pop_size):
            self.gradient_mutate(mutate_pop, mutate_optim, rsgan_train_images, fitness_train_images, gene=self.genes[i],
                                 optimizer=self.genes_optimizer[i],
                                 mode=mode)

        mutate_fitness_evals = []
        mutate_critics_samples = []
        for i in range(self.mutate_size):
            self.individual.load_state_dict(mutate_pop[i])
            fitness, gen_critic, gen_images = self.env.eval_worker(mode[i % mode_number], noise_mutate,
                                                                   eval_images)
            mutate_fitness_evals.append(fitness)
            mutate_critics_samples.append((gen_critic, gen_images))

        ########## CROSSOVER ############
        # Initialize crossover population
        crossover_pop = []
        crossover_optim = []

        sorted_groups = SSNE.sort_groups_by_fitness(range(self.mutate_size), mutate_fitness_evals)

        # Crossover for unselected genes with 100 percent probability
        for i in range(self.crossover_size):
            first, second, _ = sorted_groups[i % len(sorted_groups)]
            if mutate_fitness_evals[first] < mutate_fitness_evals[second]:
                first, second = second, first
            self.distilation_crossover(noise_mutate, mutate_pop[first], mutate_pop[second], mutate_optim[first],
                                       mutate_optim[second], mutate_critics_samples[first],
                                       mutate_critics_samples[second], crossover_pop, crossover_optim)

            # # test
            # parents1_eval = mutate_fitness_evals[first]
            # parents2_eval = mutate_fitness_evals[second]
            # print("parents:", first, parents1_eval, second, parents2_eval, mutate_fitness_evals)

        crossover_fitness_evals = []
        crossover_critics_samples = []
        for i in range(self.crossover_size):
            self.individual.load_state_dict(crossover_pop[i])
            crossover_fitness, crossover_gen_critic, crossover_gen_images = self.env.eval_worker('crossover',
                                                                                                 noise_crossover,
                                                                                                 eval_images)
            crossover_fitness_evals.append(crossover_fitness)
            crossover_critics_samples.append((crossover_gen_critic, crossover_gen_images))

            # # test
            # child = crossover_fitness
            # print("child:", child)

        ########## SELECTION ############
        if self.args.use_comprehensive_selection:
            if self.args.eval_parents:
                fitness_evals = parents_fitness_evals + mutate_fitness_evals + crossover_fitness_evals
            else:
                fitness_evals = mutate_fitness_evals + crossover_fitness_evals
        else:
            fitness_evals = crossover_fitness_evals
        top_n = np.argsort(fitness_evals)[-self.args.pop_size:]

        # Sync evo to pop
        self.selection_stats['total'] += 1

        selected = None
        evalimg_list = []
        for i in range(self.args.pop_size):
            index = top_n[i]

            if self.args.use_comprehensive_selection:
                if self.args.eval_parents:
                    if index >= self.args.pop_size + self.mutate_size:
                        index = index - self.mutate_size - self.args.pop_size
                        self.genes[i] = copy.deepcopy(crossover_pop[index])
                        self.genes_optimizer[i] = copy.deepcopy(crossover_optim[index])
                        evalimg_list.append(crossover_critics_samples[index][1])

                        selected = 'crossover'
                        self.selection_stats['crossover'] += 1
                    elif index >= self.args.pop_size:
                        index = index - self.args.pop_size
                        self.genes[i] = copy.deepcopy(mutate_pop[index])
                        self.genes_optimizer[i] = copy.deepcopy(mutate_optim[index])
                        evalimg_list.append(mutate_critics_samples[index][1])

                        selected = mode[index % mode_number]
                        self.selection_stats[selected] += 1
                    else:
                        self.genes[i] = copy.deepcopy(parents_pop[index])
                        self.genes_optimizer[i] = copy.deepcopy(parents_optim[index])
                        evalimg_list.append(parents_critics_samples[index][1])

                        selected = 'parents'
                        self.selection_stats['parents'] += 1

                else:
                    if index >= self.mutate_size:
                        index = index - self.mutate_size
                        self.genes[i] = copy.deepcopy(crossover_pop[index])
                        self.genes_optimizer[i] = copy.deepcopy(crossover_optim[index])
                        evalimg_list.append(crossover_critics_samples[index][1])

                        selected = 'crossover'
                        self.selection_stats['crossover'] += 1
                    else:
                        self.genes[i] = copy.deepcopy(mutate_pop[index])
                        self.genes_optimizer[i] = copy.deepcopy(mutate_optim[index])
                        evalimg_list.append(mutate_critics_samples[index][1])

                        selected = mode[index % mode_number]
                        self.selection_stats[selected] += 1

            else:
                self.genes[i] = copy.deepcopy(crossover_pop[index])
                self.genes_optimizer[i] = copy.deepcopy(crossover_optim[index])
                evalimg_list.append(crossover_critics_samples[index][1])

                selected = 'crossover'
                self.selection_stats['crossover'] += 1

        # for i in range(self.args.pop_size):
        #     index = top_n[i]
        #     if index >= pop_csize:
        #         index = index - pop_csize
        #         self.genes[i] = copy.deepcopy(mutate_pop[index])
        #         self.genes_optimizer[i] = copy.deepcopy(mutate_optim[index])
        #         evalimg_list.append(mutate_critics_samples[index][1])
        #
        #         selected = mode[index % mode_number]
        #         self.selection_stats[selected] += 1
        #
        #     else:
        #         self.genes[i] = copy.deepcopy(crossover_pop[index])
        #         self.genes_optimizer[i] = copy.deepcopy(crossover_optim[index])
        #         evalimg_list.append(crossover_critics_samples[index][1])
        #
        #         selected = 'crossover'
        #         self.selection_stats['crossover'] += 1

        evalimg = torch.cat(evalimg_list, dim=0)
        shuffle_ids = torch.randperm(evalimg.size()[0])
        evalimg = evalimg[shuffle_ids]

        if self.args.eval_criteria != 'operator_test':
            # Migration Tracker
            select_rate_dict = {g_loss: self.selection_stats[g_loss] / self.selection_stats['total'] for g_loss in mode}
            select_times_dict = {g_loss: self.selection_stats[g_loss] for g_loss in mode}
            if self.crossover_size != 0:
                select_rate_dict.update(
                    {'crossover': self.selection_stats['crossover'] / self.selection_stats['total']})
                select_times_dict.update({'crossover': self.selection_stats['crossover']})
            if self.args.eval_parents:
                select_rate_dict.update({'parents': self.selection_stats['parents'] / self.selection_stats['total']})
                select_times_dict.update({'parents': self.selection_stats['parents']})
            self.writer.add_scalars('select_rate', select_rate_dict, gen)
            self.writer.add_scalars('select_times', select_times_dict, gen)

            # self.writer.add_scalars('Fq', Fq, gen)
            # self.writer.add_scalars('Fd', Fd, gen)
            # self.writer.add_scalars('Fitness', F, gen)
            # self.writer.add_scalars('Fq_minmax', Fq_minmax, gen)
            # self.writer.add_scalars('Fd_minmax', Fd_minmax, gen)
            # self.writer.add_scalars('Fitness_minmax', F_minmax, gen)

        return evalimg, selected
