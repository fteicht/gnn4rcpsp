from .graph import Graph
import torch
from torch_geometric.nn import TransformerConv, GINEConv, Sequential
from math import sqrt

class Transformer5(torch.nn.Module):
    N_HIDDEN = 64
    NODE_INPUT_DIM = Graph.NODE_INPUT_DIM
    EDGE_FEATURE_DIM = Graph.EDGE_FEATURE_DIM
    def __init__(self):
        super(Transformer5, self).__init__()
        self.conv1 = TransformerConv(in_channels=self.NODE_INPUT_DIM,
                                     out_channels=self.N_HIDDEN,
                                     edge_dim=self.EDGE_FEATURE_DIM,
                                     )
        # self.conv2 = TransformerConv(in_channels=N_HIDDEN,
        #                              out_channels=N_HIDDEN,
        #                              edge_dim=EDGE_FEATURE_DIM)
        self.conv2 = Sequential('x, edge_index, edge_attr', [
            (TransformerConv(in_channels=self.N_HIDDEN,
                             out_channels=self.N_HIDDEN,
                             edge_dim=self.EDGE_FEATURE_DIM,
                             ), 'x, edge_index, edge_attr -> x')
            for _ in range(5)
        ])
    
        self.linear = torch.nn.Linear(self.N_HIDDEN, 1)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = (x + 1.).log()
        edge_attr = (edge_attr + 1.).log()

        x = self.conv1(x, edge_index, edge_attr)
        assert not x.isnan().any()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.linear(x)

        return x


class ResTransformer(torch.nn.Module):
    # N_HIDDEN = 128
    N_HIDDEN = 256
    N_BLOCKS = 15
    N_BINARY = 8
    # N_BINARY = 1
    NODE_INPUT_DIM = Graph.NODE_INPUT_DIM
    EDGE_FEATURE_DIM = Graph.EDGE_FEATURE_DIM
    transformer_options = dict(
    #     heads=8,
    #     concat=False,
    )
    def __init__(self):
        super(ResTransformer, self).__init__()
        self.conv1 = TransformerConv(in_channels=self.NODE_INPUT_DIM,
                                     out_channels=self.N_HIDDEN,
                                     edge_dim=self.EDGE_FEATURE_DIM,
                                     **self.transformer_options)

        residual_block_maker = lambda: (
            Sequential('x, edge_index, edge_attr', [
                (lambda x: torch.relu(x / x.std(-1, keepdim=True).detach()), 'x -> x2'),
                (TransformerConv(in_channels=self.N_HIDDEN,
                                  out_channels=self.N_HIDDEN,
                                  edge_dim=self.EDGE_FEATURE_DIM,
                                  **self.transformer_options), 'x2, edge_index, edge_attr -> y'),
                (lambda x, y: (x + y), 'x, y -> x'),
            ]))
        self.conv2 = Sequential('x, edge_index, edge_attr', [
            (residual_block_maker(), 'x, edge_index, edge_attr -> x')
            for _ in range(self.N_BLOCKS)
        ])
        
        # TODO: exponentiation? normalization by constant?
        self.linear = torch.nn.Linear(self.N_HIDDEN, self.N_BINARY)
        self.scale = torch.nn.Parameter(torch.tensor([2**i for i in range(self.N_BINARY)]), requires_grad=False)
        # self.scale = torch.nn.Parameter(torch.tensor([100]), requires_grad=False)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = (x + 1.).log()
        edge_attr = (edge_attr + 1.).log()

        x = self.conv1(x, edge_index, edge_attr)
        assert not x.isnan().any()
        x = self.conv2(x, edge_index, edge_attr)
        x = x / sqrt(self.N_BLOCKS) # TODO: performance improves by removing this line ??
        x = self.linear(x)

        # Build output from binary representation: e.g. [1,0,0] -> [1], [0,1,0] -> [2], [1,0,1] -> [5], ...
        # This scale & sum does: x[..., 0] * 2^0 + x[..., 1] * 2^1 + x[..., 2] * 2^2 + ... + x[..., 9]  * 2^9
        x = (x * self.scale).sum(-1, keepdim=True)

        return x


class ResGINE(torch.nn.Module):
    N_HIDDEN = 128
    N_BLOCKS = 15
    N_BINARY = 10
    NODE_INPUT_DIM = Graph.NODE_INPUT_DIM
    EDGE_FEATURE_DIM = Graph.EDGE_FEATURE_DIM
    def __init__(self):
        super(ResGINE, self).__init__()
        # self.conv1 = TransformerConv(in_channels=self.NODE_INPUT_DIM,
        #                              out_channels=self.N_HIDDEN,
        #                              edge_dim=self.EDGE_FEATURE_DIM,
        #                              )
        self.embed_node = torch.nn.Linear(self.NODE_INPUT_DIM, self.N_HIDDEN)
        # self.embed_edge = torch.nn.Linear(self.EDGE_FEATURE_DIM, self.N_HIDDEN)

        residual_block_maker = lambda: (
            Sequential('x, edge_index, edge_attr', [
                (lambda x: x / x.std(-1, keepdim=True).detach(), 'x -> x2'), # normalize after residual
                (torch.nn.Linear(self.EDGE_FEATURE_DIM, self.N_HIDDEN), 'edge_attr -> edge_attr'), # embed into N_HIDDEN size
                (GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(self.N_HIDDEN, self.N_HIDDEN),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.N_HIDDEN, self.N_HIDDEN),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.N_HIDDEN, self.N_HIDDEN),
                    )
                        ), 'x2, edge_index, edge_attr -> y'),
                (lambda x: x / x.std(-1, keepdim=True).detach(), 'y -> y'), # normalize AFTER non-linear transform
                (lambda x, y: (x + y), 'x, y -> x'), # Residual connection
            ]))
        self.conv = Sequential('x, edge_index, edge_attr', [
            (residual_block_maker(), 'x, edge_index, edge_attr -> x')
            for _ in range(self.N_BLOCKS)
        ])
        
        # TODO: exponentiation? normalization by constant?
        self.linear = torch.nn.Linear(self.N_HIDDEN, self.N_BINARY)
        self.scale = torch.nn.Parameter(torch.tensor([2**i for i in range(self.N_BINARY)]), requires_grad=False)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = (x + 1.).log()
        edge_attr = (edge_attr + 1.).log()

        x = self.embed_node(x)
        # edge_attr = self.embed_edge(edge_attr)
        # edge_attr = edge_attr / edge_attr.std(-1, keepdim=True).detach() # normalize
        x = self.conv(x, edge_index, edge_attr)
        x = x / sqrt(self.N_BLOCKS)
        x = self.linear(x)

        # Build output from binary representation: e.g. [1,0,0] -> [1], [0,1,0] -> [2], [1,0,1] -> [5], ...
        # This scale & sum does: x[..., 0] * 2^0 + x[..., 1] * 2^1 + x[..., 2] * 2^2 + ... + x[..., 9]  * 2^9
        x = (x * self.scale).sum(-1, keepdim=True)

        return x