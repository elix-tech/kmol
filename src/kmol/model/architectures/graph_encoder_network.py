from typing import Dict, Any, List

import torch

from ...core.helpers import SuperFactory
from ..layers import GraphormerGraphEncoder

from .abstract_network import AbstractNetwork


class GraphormerEncoderNetwork(AbstractNetwork):
    def __init__(
        self,
        pre_layernorm: bool = False,
        num_atoms: int = 512 * 9,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_edges: int = 1024 * 3,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        edge_type: str = "multi_hop",
        multi_hop_max_dist: int = 5,
        # max_nodes: int = 128,
        num_classes: int = 1,
        remove_head: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        encoder_ffn_embed_dim: int = 4096,
        encoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        encoder_embed_dim: int = 1024,
        share_encoder_input_output_embed: bool = False,
        apply_graphormer_init: bool = False,
        activation: str = "torch.nn.GELU",
        encoder_normalize_before: bool = True,
    ):
        super().__init__()
        # self.max_nodes = max_nodes
        self.out_features = encoder_embed_dim

        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            # >
            num_encoder_layers=encoder_layers,
            embedding_dim=encoder_embed_dim,
            ffn_embedding_dim=encoder_ffn_embed_dim,
            num_attention_heads=encoder_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            encoder_normalize_before=encoder_normalize_before,
            pre_layernorm=pre_layernorm,
            apply_graphormer_init=apply_graphormer_init,
            activation=activation,
        )

        self.share_input_output_embed = share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not remove_head

        self.masked_lm_pooler = torch.nn.Linear(encoder_embed_dim, encoder_embed_dim)

        self.lm_head_transform_weight = torch.nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.activation = SuperFactory.reflect(activation)()
        self.layer_norm = torch.nn.LayerNorm(encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = torch.nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = torch.nn.Linear(encoder_embed_dim, num_classes, bias=False)
            else:
                raise NotImplementedError

    def get_requirements(self) -> List[str]:
        return ["graph"]

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = torch.nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, data: Dict[str, Any], perturb=None, masked_tokens=None, **unused):
        data = data[self.get_requirements()[0]]
        # x = data.x.float()
        inner_states, graph_rep = self.graph_encoder(data)

        x = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            x = torch.functional.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x[:, 0, :]
