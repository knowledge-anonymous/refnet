from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

import torch_scatter

from .layers import Dense, ScaleShift, GetItem, MLP


class EquivariantDecoderBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim, activation=F.silu, final_activation=None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Vector communication layer
        self.vector_comm = Dense(in_dim, out_dim, activation=None, bias=False)

        # Atom-wise communication layer
        self.atom_wise_comm = nn.Sequential(
            Dense(in_dim + out_dim, hidden_dim, activation=activation),
            Dense(hidden_dim, out_dim * 2, activation=None),
        )

        self.final_activation = final_activation

    def forward(self, scalar, vector):
        # Pass vector through the vector communication layer
        comm_vector = self.vector_comm(vector)

        # Concatenate scalar and norm of the communication vector
        reps = torch.cat([scalar, torch.norm(comm_vector, dim=-2)], dim=-1)

        # Pass the context through the atom-wise communication layer
        x = self.atom_wise_comm(reps)

        # Split the output
        scalar_out, x = torch.split(x, [self.out_dim, self.out_dim], dim=-1)

        # Apply final activation if present
        final_scalar_out = (
            self.final_activation(scalar_out) if self.final_activation else scalar_out
        )

        return final_scalar_out, x.unsqueeze(-2) * comm_vector


class Decoder(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.
    """

    def __init__(
        self,
        in_dims,
        out_dims=1,
        n_layers=2,
        hidden_dims=None,
        activation=F.silu,
        property="y",
        mean=None,
        stddev=None,
        atomref=None,
        decoder_nn=None,
        pos_cont=False,
        return_vector=None,
        aggr_fn: Optional[str] = "sum",
    ):
        super(Decoder, self).__init__()

        self.init_variables(
            in_dims,
            out_dims,
            n_layers,
            hidden_dims,
            activation,
            property,
            mean,
            stddev,
            atomref,
            decoder_nn,
            pos_cont,
            return_vector,
            aggr_fn,
        )

        self.init_output_network(
            decoder_nn, in_dims, hidden_dims, out_dims, n_layers, activation
        )

        self.aggregation_mode = aggr_fn

    def init_variables(
        self,
        in_dims,
        out_dims,
        n_layers,
        hidden_dims,
        activation,
        property,
        mean,
        stddev,
        atomref,
        decoder_nn,
        pos_cont,
        return_vector,
        aggr_fn,
    ):
        self.return_vector = return_vector
        self.n_layers = n_layers
        self.property = property
        self.pos_cont = pos_cont

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        self.init_standardization_layer(mean, stddev)

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(atomref.type(torch.float32))
        else:
            self.atomref = None

        self.equivariant = False

    def init_output_network(
        self, decoder_nn, in_dims, hidden_dims, out_dims, n_layers, activation
    ):
        if decoder_nn is None:
            self.out_net = nn.Sequential(
                GetItem("s"),
                MLP(
                    [in_dims, hidden_dims, out_dims],
                    n_layers=n_layers,
                    activation=activation,
                ),
            )
        elif decoder_nn == "derived":
            if hidden_dims is None:
                hidden_dims = in_dims
            self.out_net = nn.ModuleList(
                [
                    EquivariantDecoderBlock(
                        in_dim=in_dims,
                        out_dim=hidden_dims,
                        hidden_dim=hidden_dims,
                        activation=activation,
                        final_activation=activation,
                    ),
                    EquivariantDecoderBlock(
                        in_dim=hidden_dims,
                        out_dim=1,
                        hidden_dim=hidden_dims,
                        activation=activation,
                    ),
                ]
            )
            self.derived = True
        else:
            self.out_net = decoder_nn

    def init_standardization_layer(self, mean, stddev):
        if mean is not None and stddev is not None:
            self.standardize = ScaleShift(mean, stddev)
        else:
            self.standardize = nn.Identity()


    def forward_derived(self, inputs, result):
        s, V = inputs.s, inputs.V

        for l in self.out_net:
            l0, l1 = l(s, V)

        if self.pos_cont:
            atomic_dipoles = torch.squeeze(V, -1)
            charges = s
            dipole_offsets = inputs.pos * charges

            yi = atomic_dipoles + dipole_offsets
        else:
            yi = s

        if self.return_vector:
            result[self.return_vector] = V

        result[self.property] = self.standardize(yi)
        return result

    def forward_default(self, inputs, result):
        yi = self.out_net(inputs)
        yi = self.standardize(yi)
        result[self.property] = yi
        return result

    def forward(self, inputs):
        atoms = inputs.z
        result = {}

        if self.derived:
            result = self.forward_derived(inputs, result)
        else:
            result = self.forward_default(inputs, result)

        if self.atomref is not None:
            result[self.property] = self.atomref(atoms) + result[self.property]

        if self.aggregation_mode is not None:
            result[self.property] = torch_scatter.scatter(
                result[self.property], inputs.batch, dim=0, reduce=self.aggregation_mode
            )

        return result
