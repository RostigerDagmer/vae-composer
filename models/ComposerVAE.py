import torch
from argparse import ArgumentParser
import os
from pathlib import Path
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
import torchvision
import pytorch_lightning as pl
from .types_ import *
from utils.logging import *


class ComposerVAE(pl.LightningModule):
    class MeasureCell(nn.Module):
        ''' Auto Encoder that encodes a batch of measures for a single track into my and sig latent vectors'''

        def __init__(self):
            super(ComposerVAE.MeasureCell, self).__init__()
            conv_1xnx1 = lambda n, s: torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(n, 1),
                                                      stride=(s, 1))
            self.conv_1x3x1 = conv_1xnx1(3, 2)  # needs to be stacked twice
            self.bn_1 = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.leaky_relu = torch.nn.functional.leaky_relu_
            self.conv_1x6x1 = conv_1xnx1(6, 3)  # needs to be stacked twice
            self.conv_1x12x1 = conv_1xnx1(12, 3)  # needs to be stacked twice
            # first spatial convolution combines a,s,d to single channel
            self.conv_3x5x5 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(1, 1))
            # convolution for first branch looking for melody correlations
            self.conv_1x3x3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 2))
            self.bn_2 = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.conv_1x5x5 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 2))

            self.encoder = nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 1), stride=(2, 1)),
                torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 1), stride=(2, 1)),
                torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(6, 1), stride=(3, 1)),
                torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(12, 1), stride=(3, 1)),
                torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.LeakyReLU(),
                # end of only temporal convolution
                torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
                torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 2)),
                torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 2)),
                torch.nn.LeakyReLU()
            )

            self.fc_mu = nn.Linear(315, 315)
            self.fc_var = nn.Linear(315, 315)
            self.encode_shape = None  # torch.Size(315)
            self.decoder_input = nn.Linear(315, 315)

            self.decoder = nn.Sequential(

                torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 2)),
                torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 2)),
                torch.nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
                torch.nn.LeakyReLU(),

                torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(12, 1), stride=(3, 1)),
                torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(6, 1), stride=(3, 1)),
                torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 1), stride=(2, 1)),
                torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 1), stride=(2, 1)),
                torch.nn.LeakyReLU()
            )

            self.final_layer = nn.Sequential(
                nn.Tanh()
            )

            # before (measures, tracks, attributes, x, y)
            # batch shape: (songs, measures, tracks, attributes, x, y)
            # self.conv_1x3x1 = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 1, 1), stride=2)
            # "Chord filters"

        def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
            """
            Reparameterization trick to sample from N(mu, var) from
            N(0,1).
            :param mu: (Tensor) Mean of the latent Gaussian [B x D]
            :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
            :return: (Tensor) [B x D]
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self, input: torch.tensor, **kwargs):
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
            return [self.decode(z), input, z, mu, log_var]

        def decode(self, z):
            result = self.decoder_input(z)
            result = result.view(self.encode_shape)
            result = self.decoder(result)
            result = self.final_layer(result)
            return result

        def encode(self, m):
            result = self.encoder(m)
            self.encode_shape = result.shape
            result = torch.flatten(result, start_dim=1)
            mu = self.fc_mu(result)
            log_var = self.fc_var(result)
            print(f'mu: {mu.shape}; var: {log_var.shape};')
            # 315 x 1 vectors of latent space
            return [mu, log_var]

        def old_encode(self, m):
            assert m.ndim == 4  # make sure m is a batch of a * x * y measures

            # first some convolutions along the temporal(x)-Axis
            last = log_volume(m)

            m = self.conv_1x3x1(m)  # |_|_|_| -> |_| = >  |_|_|_| -> |_|

            last = log_volume(m, last)

            m = self.bn_1(m)
            m = self.conv_1x3x1(m)  # |_|_|_| -> |_| = >  |_|_|_|_|_|_| -> |_|

            last = log_volume(m, last)

            m = self.bn_1(m)
            m = self.conv_1x6x1(m)  # |_|_|_|_|_|_| -> |_|

            last = log_volume(m, last)

            m = self.conv_1x12x1(m)  # |1|2|3|4|5|6|7|8|9|10|11|12| -> |_|

            last = log_volume(m, last)

            img_grid = torchvision.utils.make_grid(m, nrow=1)
            print(f'max: {torch.max(img_grid)}')
            plt.imshow(img_grid.transpose(0, 2).detach())
            plt.show()

            ######### BRANCH 1 #############

            # next look for some correlations in the 2d plane
            # make sure to have extra convolutions using 'chord filters'
            #                                                _ _ _ _ _
            #                                                        /|  _
            #                                                      /|/|  /|
            #                                          _ _ _ _ _ /|/|/|       correlate Notes in normalized temporal space
            #                                         |_|_|_|_|_|/|/|/|
            #                                         |_|_|_|_|_|/|/|/|
            #                                       # |_|_|_|_|_|/|/|/
            m_branch1 = self.bn_1(m)                # |_|_|_|_|_|/|/
            m_branch1 = self.conv_3x5x5(m_branch1)  # |_|_|_|_|_|/

            last = log_volume(m_branch1, last)

            m_branch1 = self.bn_2(m_branch1)
            m_branch1 = self.conv_1x3x3(m_branch1)  # first convolution over spots in note space

            last = log_volume(m_branch1, last)

            m_branch1 = self.conv_1x5x5(m_branch1)  # second convolution over spots in note space

            last = log_volume(m_branch1, last)

            m_branch1 = self.conv_1x5x5(m_branch1)

            last = log_volume(m_branch1, last)

            m_branch1 = self.leaky_relu(m_branch1)

            plt.imshow(m_branch1[0].detach().squeeze(0))
            plt.show()

            ########## branch_2 ##########

            m_branch2 = self.bn_1(m)

            return m_branch1

    class MergerCell(nn.Module):
        ''' AutoencoderCell that merges sampled latent vectors of MeasureCell's into mu and sig latent vectors '''

        def __init__(self, inchannels=10):
            super(ComposerVAE.MergerCell, self).__init__()
            self.conv_3x3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3))
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(2),
                nn.LeakyReLU()
            )
            self.fc_mu = nn.Linear(64, 64)
            self.fc_var = nn.Linear(64, 64)
            self.decoder_input = nn.Linear(64, 64)
            self.encode_shape = None

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(3),
                nn.LeakyReLU()
            )
            self.final_layer = nn.Tanh()

        def forward(self, t):
            # assert t.ndim == 4 # make sure that t is a batch of s * t * m * a * x * y
            t = t.view(-1, 3, 7, 15)  # reshape sample vector to 2d image batch
            mu, log_var = self.encode(t)
            z = self.reparameterize(mu, log_var)
            return [self.decode(z), input, z, mu, log_var]
            '''
            print(t.shape)

            return self.encode(t)
            '''

        def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
            """
            Reparameterization trick to sample from N(mu, var) from
            N(0,1).
            :param mu: (Tensor) Mean of the latent Gaussian [B x D]
            :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
            :return: (Tensor) [B x D]
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def decode(self, z):
            result = self.decoder_input(z)
            result = result.view(self.encode_shape)
            result = self.decoder(result)
            result = self.final_layer(result)
            return result

        def encode(self, t):
            batch, c, w, h = t.shape
            result = self.encoder(t)
            self.encode_shape = result.shape
            result = result.view(batch, -1)
            mu = self.fc_mu(result)
            log_var = self.fc_var(result)
            return [mu, log_var]

    def __init__(self, max_tracks=10):
        super(ComposerVAE, self).__init__()
        self.merger_cell = self.MergerCell()
        self.measure_cell = self.MeasureCell()

    def forward(self, s):
        # takes a minibatch of m * a * x * y
        # measures: m
        # tracks: t
        # attributes: a
        # length: x
        # width: y

        # -> measures for different tracks can be flattened because we treat tracks of songs like songs

        # pass each track through measure cell
        # s = s.transpose(1, 2) #switch measure and track dimension
        mc_s = self.measure_cell.encode(s)  # sample of measure cell (batch of tracks)
        print(mc_s)
        # containing: [reconstruction, input, sampled_vector z, mu, log_var]
        f_s = self.merger_cell(mc_s[2])[
            0]  # merge samples for different tracks into one latent vector describing the song
        print(f_s.shape)

    @staticmethod

    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover

        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)
        # network params

        parser.add_argument('--in_features', default=128 * 96, type=int)
        #parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--out_features', default=64, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        # data
        cwd = Path(os.getcwd())
        root_dir = cwd.parent / "datasets" / "midiset"
        parser.add_argument('--data_root', default=root_dir, type=str)
        parser.add_argument('--num_workers', default=8, type=int)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        return parser

'''
class InfoVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha: float = -0.5,
                 beta: float = 5.0,
                 reg_weight: int = 100,
                 kernel_type: str = 'imq',
                 latent_var: float = 2.,
                 **kwargs) -> None:
        super(InfoVAE, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        assert alpha <= 0, 'alpha must be negative or zero.'

        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, z, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]

        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = self.beta * recons_loss + \
               (1. - self.alpha) * kld_weight * kld_loss + \
               (self.alpha + self.reg_weight - 1.)/bias_corr * mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'MMD': mmd_loss, 'KLD':-kld_loss}

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result


    def compute_rbf(self,
                    x1: Tensor,
                    x2: Tensor,
                    eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                               x1: Tensor,
                               x2: Tensor,
                               eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
'''
