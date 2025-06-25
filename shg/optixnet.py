import torch
from torch import nn
from torch_scatter import scatter
from transformer import ComformerConv, ComformerConvEqui, Output_block
from utils import RBFExpansion


class OptixNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        embsize = 256
        self.atom_embedding = nn.Linear(
            92, embsize
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=512,
            ),
            nn.Linear(512, embsize),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                for _ in range(4)
            ]
        )

        self.equi_update = ComformerConvEqui(in_channels=embsize, edge_dim=embsize)

        self.output_block = Output_block()


    def forward(self, data) -> torch.Tensor:
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)

        for i in range(len(self.att_layers)):
            node_features = self.att_layers[i](node_features, data.edge_index, edge_features)

        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
        
        outputs = self.output_block(crystal_features)

        return outputs