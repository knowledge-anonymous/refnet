from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.nn import Parameter
from torch_scatter import scatter

from .layers import MLP, Dense, LeakyCosineCutoff, _VDropout, safe_norm, GetItem, get_geometry
from .torsion import stable_torsion_wrapper
from .utils import INV_SQRT_3, _init, _merge, _split, basis, get_logger
from . import decoder as dec

log = get_logger(__name__)


class RefNet(nn.Module):
    """
        The RefNet class implements a vectorial equivariant message passing neural network for
        scalable molecular representation learning.

        The model follows an encoder-decoder structure where the encoder layers learn the geometric features'
        latent representation, and the decoder layers decode the learned latent representations for downstream
        tasks on either invariant or equivariant targets. This class implements the encoder layers.

        Mathematically, the inter-atomic interactions are described as:

        \\[ IA(s, V, r, \vec{d}) = s'_i, V'_i = (s_i, V_i) + \sum_{j \in \mathcal{N}_i}{e(s_j, v_j, {r}_{ij}, \vec{\beta}_{ij})} \\]

        The interaction function is formalized as:

        \\[ e_s(s_j, v_j, {r}_{ij}, \vec{\beta}_{ij}) = \phi_s(h_j) \odot \eta_s(r_{ij}) \\
            e_v(s_j, v_j, {r}_{ij}, \vec{\beta}_{ij}) = VA(\phi_b(\vec{\beta}_{ij}) \odot \phi_d(h_j) \odot \eta_d(r_{ij}) + \phi_v(h_j) \odot \eta_r(r_{ij})) \\]

        The atom-wise blocks compute the interaction between invariant and equivariant representations and channel-wise updates:

        \\[ AW(s, V) = \{(\phi_m(||\phi_{vu}(V)|| \cup s)), (VA(\phi_{vu}(V) \odot \phi_v(||\phi_{vu}(V)|| \cup s_i))\} \\]


    Args:
        hidden_dim (int, optional): The dimension of the hidden layers in the network. Defaults to 128.
        num_encoder (int, optional): The number of encoders to use in the network. Defaults to 8.
        num_rbf (int, optional): The number of radial basis functions (RBF). Defaults to 20.
        cutoff (float, optional): The cutoff distance for atomic interactions. Defaults to 5.0.
        r_basis (str, optional): The type of radial basis function. Defaults to "BesselBasis".
        activation (str, optional): The activation function to use in the network. Defaults to "swish".
        max_z (int, optional): The maximum atomic number that the network can handle. Defaults to 100.
        weight_init (str, optional): The weight initialization strategy. Defaults to "xavier_uniform".
        bias_init (str, optional): The bias initialization strategy. Defaults to "zeros".
        vector_embed_dropout (float, optional): Dropout rate for the vector embeddings. Defaults to 0.0.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder: int = 8,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        r_basis: str = "BesselBasis",
        activation="swish",
        max_z: int = 100,
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
        vector_embed_dropout=0.0,
        **kwargs,
    ):
        """
        Initialize RefNet.
        """
        super(RefNet, self).__init__()
        self.vector_embed_dropout = vector_embed_dropout
        seed_everything(42, workers=True)
        self.eps = 1e-8
        self.hidden_dim = hidden_dim
        self.num_encoder = num_encoder
        self.vector_embed_dropout = _VDropout(vector_embed_dropout)
        self.vector_alpha = 1.0

        self.omega = LeakyCosineCutoff(cutoff, 0)
        self.nonlinear_bias = nn.ModuleList(
            [
                nn.ParameterList(
                    [
                        Parameter(torch.zeros(hidden_dim))
                        for _ in range(self.num_encoder)
                    ]
                )
                for _ in range(2)
            ]
        )  # message_bias_non, update_bias_non

        radial_basis = basis(r_basis)

        self.chi = radial_basis(cutoff=cutoff, n_rbf=num_rbf)
        self.embedding = nn.Embedding(max_z, hidden_dim, padding_idx=0)
        self.vector_embedding = nn.Linear(hidden_dim, hidden_dim * 3)

        activation, bias_init, weight_init = _init(activation, bias_init, weight_init)
        DenseInit = partial(Dense, weight_init=weight_init, bias=bias_init)
        MLPInit = partial(MLP, weight_init=weight_init, bias=bias_init)

        self.phi_chi_gamma = DenseInit(num_rbf, self.num_encoder * 3 * self.hidden_dim)

        self.phi_b = DenseInit(3, hidden_dim, bias=False)

        self.inter_atomic_comm = nn.ModuleList(
            [
                MLPInit(
                    hidden_dims=[hidden_dim, hidden_dim, 3 * hidden_dim],
                    activation=activation,
                )
                for _ in range(self.num_encoder)
            ]
        )
        self.atom_wise_comm = nn.ModuleList(
            [
                MLPInit(
                    hidden_dims=[
                        hidden_dim + hidden_dim,
                        hidden_dim,
                        hidden_dim + hidden_dim,
                    ],
                    activation=activation,
                )
                for _ in range(self.num_encoder)
            ]
        )
        self.vector_update = nn.ModuleList(
            [
                DenseInit(
                    hidden_dim,
                    hidden_dim,
                    bias=False,
                )
                for _ in range(self.num_encoder)
            ]
        )

        self.message_sign = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.n_interactions)
            ]
        )
        self.embed_sign = nn.Embedding(2, hidden_dim)

    def forward(self, batch_data):
        """
        The forward function implements the process of the forward pass through the RefNet model.
        It takes as input a batch of molecule data and processes it to return the scalar and vector representations.

        The process can be described as follows:

        1. The input is passed through the initialization layers, where the geometric features'
           latent representation is learned.

        2. These latent representations are then passed to the inter-atomic interaction function
           which encapsulates inter-atomic interactions by integrating both equivariant direction vectors
           and invariant distance filters, encoded using radial basis functions.

        3. The outputs of the interaction function are then processed by atom-wise blocks,
           which compute the interaction between invariant and equivariant representations and perform channel-wise updates.

        4. The results are then passed through the decoder layers where the learned latent representations
           are decoded for downstream tasks on either invariant or equivariant targets.

        Args:
        batch_data (Data): Batch of molecule data.

        Returns:
        Data: Batch that includes scalar and vector representations.
        """
        atom, pos, batch = batch_data.z.long(), batch_data.pos, batch_data.batch
        rij, d_ij = batch_data.rij, batch_data.dir_ij

        edge_index = batch_data.edge_index
        j, i = edge_index

        t_ij = torch.cross(pos[j], pos[i])
        o_ij = torch.cross(d_ij, t_ij)
        edge_vectors = torch.stack([d_ij, t_ij, o_ij], dim=-1)
        beta_ij = self.phi_b(edge_vectors)

        torsion, sign, selected = stable_torsion_wrapper(batch_data, torsion_type=self.torsion_type)

        choose_mask = sign == 0
        # update negative sign to 0
        sign[sign == -1] = 0
        # embed sign
        sign_embed = self.embed_sign(sign.to(torch.long))
        # clear sign for non-torsion nodes
        sign_embed[choose_mask] = 0.0

        phi_r = self.phi_chi_gamma(self.chi(rij) * self.omega(rij).unsqueeze(-1))
        phi_r = torch.split(phi_r, 3 * self.hidden_dim, dim=-1)

        s = self.embedding(atom)
        spherical = self.vector_embedding(s).reshape(
            s.shape[0], -1, s.shape[-1]
        )

        V = self.vector_embed_dropout(
            torch.stack(
                (
                    torch.sin(spherical[:, 0, :]) * torch.cos(spherical[:, 1, :]),
                    torch.sin(spherical[:, 0, :]) * torch.sin(spherical[:, 1, :]),
                    torch.cos(spherical[:, 0, :]),
            ), dim=1)
            * spherical[:, 2, :].unsqueeze(1)
            * self.vector_alpha
        )

        for l_e in range(self.num_encoder):
            sign_v2 = self.message_sign[l_e](sign_embed)
            sign_v2[choose_mask] = 0

            phi_s_h_j = phi_r[l_e] * self.inter_atomic_comm[l_e](s + sign_v2)[j]
            phi_b_d_v, phi_d, phi_v = torch.split(phi_s_h_j, self.hidden_dim, dim=-1)
            vec_update = phi_d.unsqueeze(-2) * beta_ij + (
                    phi_v.unsqueeze(-2) * INV_SQRT_3 * V[j]
            )
            merged = _merge(phi_b_d_v, vec_update)
            merged_sum = scatter(merged, i, dim=0, reduce="sum", dim_size=s.shape[0])
            phi_b_d_v, vec_update = _split(merged_sum, vec_update.shape[-1])

            norm = safe_norm(vec_update, dim=1, keepdim=True, eps=self.eps)
            norm = norm + self.nonlinear_bias[0][l_e]
            act = F.sigmoid(norm)
            V = V + (vec_update * act)
            s = s + phi_b_d_v

            updated_v = self.vector_update[l_e](V)
            v_norm = safe_norm(updated_v, dim=-2, eps=self.eps)
            reps = torch.cat([s, v_norm], dim=-1)
            phi_vm = self.atom_wise_comm[l_e](reps)

            norm = safe_norm(updated_v, dim=1, keepdim=True, eps=self.eps)
            norm = norm + self.nonlinear_bias[1][l_e]
            act = F.sigmoid(norm)
            v_prime =  updated_v * act
            v_prime = phi_vm[..., self.hidden_dim:].unsqueeze(1) * v_prime
            s, V = s + phi_vm[..., : self.hidden_dim], V + v_prime

        batch.s, batch.V = s, V
        return batch


class RefNetWrapper(nn.Module):
    """
     RefNetWrapper, a model that encapsulates the RefNet model for end-to-end prediction.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder: int = 8,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        r_basis: str = "BesselBasis",
        activation="swish",
        max_z: int = 100,
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
        vector_embed_dropout=0.0,
        decoder=None,
        target_mean=None,
        target_std=None,
        atomref=None
    ):
        """
        Initialize RefNetWrapper.

        The wrapper consists of an encoder part, the RefNet, and a decoder part.
        The RefNet is responsible for encoding the molecular structure into a latent representation, while
        the decoder translates these representations into the target properties.

        Args:
            hidden_dim (int): Dimension of the hidden layer.
            num_encoder (int): Number of encoder layers.
            num_rbf (int): Number of Radial Basis Functions for the distance encoding.
            cutoff (float): Cutoff distance for the local environment.
            r_basis (str): Radial basis type.
            activation (str): Activation function to use. Default is "swish".
            max_z (int): Maximum atomic number that the model will be applied to.
            weight_init (str): Weight initialization method.
            bias_init (str): Bias initialization method.
            vector_embed_dropout (float): Dropout rate for the vector embedding.
            decoder (nn.Module, optional): Decoder module to use. If None, a suitable default will be chosen.
            target_mean (torch.Tensor, optional): Mean of target property. Used for de-normalizing the prediction.
            target_std (torch.Tensor, optional): Standard deviation of target property. Used for de-normalizing the prediction.
            atomref (torch.Tensor, optional): Reference single-atom properties. Used for BAC.
        """
        self.cutoff = cutoff
        self.encoder = RefNet(hidden_dim=hidden_dim, num_encoder=num_encoder, num_rbf=num_rbf, cutoff=cutoff, r_basis=r_basis, activation=activation, max_z=max_z,
                               weight_init=weight_init, bias_init=bias_init, vector_embed_dropout=vector_embed_dropout)

        if decoder is None:
            self.decoder = dec.Decoder(
                in_dims=hidden_dim,
                out_dims=1,
                num_layers=2,
                activation=activation,
                property="y",
                mean=target_mean,
                stddev=target_std,
                atomref=atomref
            )

    def forward(self, batch):
        get_geometry(batch, cutoff=self.cutoff)
        encoded = self.encoder(batch)
        output = self.decoder(encoded)
        return output
