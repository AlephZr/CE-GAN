import torch
from torch import nn, autograd


def calc_gradient_penalty(netD, real_data, fake_data, image_size, device, nc, size):
    # print("real_data: ", real_data.size(), fake_data.size())
    alpha = torch.rand(image_size, 1)
    alpha = alpha.expand(image_size, real_data.nelement() // image_size).contiguous().view(image_size, nc, size, size)
    alpha = alpha.to(device=device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device=device)
    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty


class Evaluator:
    def __init__(self, args, generator, discriminator):
        self.args = args
        self.device = args.device
        self.d_mode = args.d_loss_mode
        self.netS = generator  # Subject
        self.netE = discriminator  # Examiner

        self.ones_label = torch.ones(args.eval_size, device=args.device)
        self.zeros_label = torch.zeros(args.eval_size, device=args.device)

        self.ones_label_batch = torch.ones(args.batch_size, device=args.device)
        self.zeros_label_batch = torch.zeros(args.batch_size, device=args.device)

        # Loss function
        self.criterion_BCEL = nn.BCEWithLogitsLoss()
        self.criterion_MSE = nn.MSELoss()

        if args.eval_criteria == 'QalD':
            if self.d_mode == 'vanilla' or self.d_mode == 'nsgan':
                self.eval_worker = self.QalD_nv
            if self.d_mode == 'lsgan':
                self.eval_worker = self.QalD_lsgan
            if self.d_mode == 'wgan':
                self.eval_worker = self.QalD_wgan
        elif args.eval_criteria == 'discriminator':
            self.eval_worker = self.disc
        elif args.eval_criteria == 'operator_test':
            self.eval_worker = self.operator_test
        else:
            raise NotImplementedError('Illegal criteria！')

    def QalD_nv(self, operator_type, fake_noise, real_samples):
        # Dataset iterator
        for p in self.netE.parameters():
            p.requires_grad = True

        self.netE.zero_grad()

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()

        ### vanilla/nsgan
        D_real = self.netE(real_samples)
        D_fake = self.netE(gen_samples)
        D_critic = torch.sigmoid(D_fake).detach().cpu().numpy()
        Fq = D_critic.mean()
        errD_real = self.criterion_BCEL(D_real, self.ones_label)
        errD_fake = self.criterion_BCEL(D_fake, self.zeros_label)
        errD = errD_real + errD_fake

        gradients = autograd.grad(outputs=errD, inputs=self.netE.parameters(),
                                  grad_outputs=torch.ones(errD.size()).to(device=self.device), create_graph=True,
                                  retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad, grad])
        # Fd = -torch.log(torch.norm(allgrad) + self.args.eps).detach().cpu().numpy()
        Fd = -torch.log(torch.norm(allgrad)).detach().cpu().numpy()
        fitness = Fq + self.args.lambda_f * Fd

        return fitness, D_critic, gen_samples

    def QalD_lsgan(self, operator_type, fake_noise, real_samples):
        # Dataset iterator
        for p in self.netE.parameters():
            p.requires_grad = True

        self.netE.zero_grad()

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()

        ### lsgan
        D_real = self.netE(real_samples)
        D_fake = self.netE(gen_samples)
        D_critic = torch.sigmoid(D_fake).detach().cpu().numpy()
        Fq = D_critic.mean()
        errD_real = self.criterion_MSE(D_real, self.ones_label)
        errD_fake = self.criterion_MSE(D_fake, self.zeros_label)
        errD = errD_real + errD_fake

        gradients = autograd.grad(outputs=errD, inputs=self.netE.parameters(),
                                  grad_outputs=torch.ones(errD.size()).to(device=self.device), create_graph=True,
                                  retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad, grad])
        # Fd = -torch.log(torch.norm(allgrad) + self.args.eps).detach().cpu().numpy()
        Fd = -torch.log(torch.norm(allgrad)).detach().cpu().numpy()
        fitness = Fq + self.args.lambda_f * Fd

        return fitness, D_critic, gen_samples

    def QalD_wgan(self, operator_type, fake_noise, real_samples):
        # Dataset iterator
        for p in self.netE.parameters():
            p.requires_grad = True

        self.netE.zero_grad()

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()

        ### WGAN-GP
        D_real = self.netE(real_samples)
        D_fake = self.netE(gen_samples)
        D_critic = torch.sigmoid(D_fake).detach().cpu().numpy()
        Fq = D_critic.mean()
        errD_real = D_real.mean()
        errD_fake = D_fake.mean()
        # # Diversity fitness score
        # gradient_penalty = calc_gradient_penalty(self.netE, real_samples.detach(), gen_samples.detach(),
        #                                          self.args.eval_size, self.device, self.args.input_nc,
        #                                          self.args.crop_size)
        # errD = errD_fake - errD_real + gradient_penalty
        errD = errD_fake - errD_real

        gradients = autograd.grad(outputs=errD, inputs=self.netE.parameters(),
                                  grad_outputs=torch.ones(errD.size()).to(device=self.device), create_graph=True,
                                  retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad, grad])
        # Fd = -torch.log(torch.norm(allgrad) + self.args.eps).detach().cpu().numpy()
        Fd = -torch.log(torch.norm(allgrad)).detach().cpu().numpy()
        fitness = Fq + self.args.lambda_f * Fd

        return fitness, D_critic, gen_samples

    def disc(self, operator_type, fake_noise, real_samples):
        for p in self.netE.parameters():
            p.requires_grad = False

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()

        D_fake = self.netE(gen_samples)
        D_critic = D_fake.detach().cpu().numpy()
        fitness = D_critic.mean()

        return fitness, D_critic, gen_samples

    def operator_test(self, operator_type, fake_noise, real_samples):
        for p in self.netE.parameters():
            p.requires_grad = False

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()
        fitness, D_critic = None, None

        return fitness, D_critic, gen_samples

    def get_fitness(self, fake_samples, real_samples):
        # Dataset iterator
        for p in self.netE.parameters():
            p.requires_grad = True

        self.netE.zero_grad()

        gen_samples = fake_samples

        ### vanilla
        D_real = self.netE(real_samples)
        D_fake = self.netE(gen_samples)
        Fq = torch.sigmoid(D_fake).mean()
        errD_real = self.criterion_BCEL(D_real, self.ones_label)
        errD_fake = self.criterion_BCEL(D_fake, self.zeros_label)
        errD = errD_real + errD_fake

        gradients = autograd.grad(outputs=errD, inputs=self.netE.parameters(),
                                  grad_outputs=torch.ones(errD.size()).to(device=self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)
        for i, grad in enumerate(gradients):
            grad = grad.view(-1)
            allgrad = grad if i == 0 else torch.cat([allgrad, grad])
        Fd = -torch.log(torch.norm(allgrad))
        fitness = Fq + self.args.lambda_f * Fd

        return fitness
