from models.base import TorchModel

from utils import get_activation_function
from utils import DEVICE

import torch
import torch.nn as nn


class BaselineProjector(torch.nn.Module):

    def __init__(
            self,
            embedding_dim,
            num_types,
            num_codes,
            max_sequence_len,
            use_positions=True,
            use_log_amount=True,
            use_layernorm=True,
            eps=1e-5,
            dropout=0.0,
            aggregation_type='sum',
            initializer_range=0.02
    ):
        super().__init__()

        self._embedding_dim = embedding_dim

        self._num_types = num_types
        self._num_codes = num_codes
        self._max_sequence_len = max_sequence_len

        self._use_positions = use_positions
        self._use_log_amount = use_log_amount
        self._use_layernorm = use_layernorm
        self._eps = eps
        self._dropout = dropout
        self._aggregation_type = aggregation_type

        self._types_embedding = nn.Embedding(
            num_embeddings=self._num_types + 2,
            embedding_dim=self._embedding_dim
        )

        self._codes_embedding = nn.Embedding(
            num_embeddings=self._num_codes + 2,
            embedding_dim=self._embedding_dim
        )

        self._amount_layer = nn.Linear(
            in_features=1, out_features=self._embedding_dim
        )

        self._position_embedding = nn.Identity()
        if self._use_positions:
            self._position_embedding = nn.Embedding(
                num_embeddings=self._max_sequence_len + 1,
                embedding_dim=self._embedding_dim
            )

        self._dropout = nn.Dropout(p=self._dropout)
        self._layernorm = nn.LayerNorm(self._embedding_dim, eps=self._eps)

        self._init_weights(initializer_range)

        self._output_dim = self._embedding_dim if self._aggregation_type == 'sum' \
            else (3 + self._use_positions) * self._embedding_dim

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            embedding_dim=config['embedding_dim'],
            num_types=kwargs['num_types'],
            num_codes=kwargs['num_codes'],
            max_sequence_len=kwargs['max_sequence_len'],
            use_positions=config.get('use_positions', True),
            use_log_amount=config.get('use_log_amount', True),
            use_layernorm=config.get('use_layernorm', True),
            eps=config.get('eps', 1e-5),
            dropout=config.get('dropout', 0.0),
            aggregation_type=config.get('aggregation_type', 'sum'),
            initializer_range=config.get('initializer_range', 0.02)
        )

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        nn.init.trunc_normal_(
            self._types_embedding.weight.data,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )

        nn.init.uniform_(self._amount_layer.weight.data, a=initializer_range, b=initializer_range)
        nn.init.uniform_(self._amount_layer.bias.data, a=initializer_range, b=initializer_range)

        nn.init.trunc_normal_(
            self._codes_embedding.weight.data,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )

        if self._use_positions:
            nn.init.trunc_normal_(
                self._position_embedding.weight.data,
                std=initializer_range,
                a=-2 * initializer_range,
                b=2 * initializer_range
            )

        nn.init.ones_(self._layernorm.weight.data)
        nn.init.zeros_(self._layernorm.bias.data)

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, inputs):
        mcc_code = inputs['mcc_code']  # (all_batch_events)
        transaction_type = inputs['transaction_type']  # (all_batch_events)
        amount = inputs['amount']  # (all_batch_events)

        mcc_code_embeddings = self._dropout(self._codes_embedding(mcc_code))  # (all_batch_events, embedding_dim)
        transaction_type_embeddings = self._dropout(self._types_embedding(transaction_type))  # (all_batch_events, embedding_dim)

        if self._use_log_amount:
            amount = torch.sign(amount) * torch.log(1. + torch.abs(amount))
        amount_embeddings = self._dropout(self._amount_layer(amount.unsqueeze(-1)))

        embeddings = [mcc_code_embeddings, transaction_type_embeddings, amount_embeddings]  # (3, all_batch_events, embedding_dim)

        if self._use_positions:
            position = inputs['positions']  # (all_batch_events)
            position_embeddings = self._position_embedding(position)  # (all_batch_events, embedding_dim)
            embeddings.append(position_embeddings)

        if self._aggregation_type == 'sum':
            embedding = None
            for e in embeddings:
                if embedding is None:
                    embedding = e
                else:
                    embedding += e
        else:
            assert self._aggregation_type == 'concat'
            embedding = torch.cat(embeddings, dim=0)

        lengths = inputs['lengths']  # (batch_size)
        batch_size = lengths.shape[0]
        max_sequence_length = lengths.max().item()

        padded_embeddings = torch.zeros(
            batch_size, max_sequence_length,
            self._embedding_dim if self._aggregation_type == 'sum' else self._embedding_dim * len(embeddings),
            dtype=torch.float, device=DEVICE
        )  # (batch_size, max_seq_len, emb_dim)

        mask = torch.arange(
            end=max_sequence_length,
            device=DEVICE
        )[None].tile([batch_size, 1]) < lengths[:, None]  # (batch_size, max_seq_len)

        padded_embeddings[mask] = embedding
        if self._use_layernorm:
            padded_embeddings = self._layernorm(padded_embeddings)

        return padded_embeddings, mask


class BaselineEncoder(torch.nn.Module):

    def __init__(
            self,
            hidden_size,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            input_dim=None,
            output_dim=None,
            user_cls_only=False,
            initializer_range=0.02
    ):
        super().__init__()

        self._input_projection = nn.Identity()
        if input_dim is not None:
            self._input_projection = nn.Linear(input_dim, hidden_size)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._output_projection = nn.Identity()
        if output_dim is not None:
            self._output_projection = nn.Linear(hidden_size, output_dim)

        self._user_cls_only = user_cls_only

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            hidden_size=config['hidden_size'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['hidden_size']),
            dropout=config.get('dropout', 0.0),
            activation=config.get('activation', 'relu'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            input_dim=kwargs['input_dim'],
            output_dim=config.get('output_dim', None),
            user_cls_only=config.get('user_cls_only', False),
            initializer_range=config.get('initializer_range', 0.02)
        )

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            else:
                raise ValueError(f'Unknown transformer weight: {key}')

    def forward(self, embeddings, attention_mask):
        embeddings = self._input_projection(embeddings)  # (batch_size, seq_len, emb_dim)
        embeddings = self._encoder(
            src=embeddings,
            src_key_padding_mask=~attention_mask
        )  # (batch_size, seq_len, emb_dim)
        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, output_emb_dim)
        if self._user_cls_only:
            embeddings = embeddings[:, 0, :]  # (batch_size, output_emb_dim)
            attention_mask = attention_mask[:, 0]  # (batch_size)
        return embeddings, attention_mask


class BaselineModel(TorchModel, config_name='baseline'):

    def __init__(self, projector, encoder):
        super().__init__()
        self._projector = projector
        self._encoder = encoder

    @classmethod
    def create_from_config(cls, config, **kwargs):
        projector = BaselineProjector.create_from_config(config['projector'], **kwargs)
        encoder = BaselineEncoder.create_from_config(
            config['encoder'],
            input_dim=projector.output_dim,
            **kwargs
        )
        return cls(projector, encoder)

    def forward(self, inputs):
        embeddings, mask = self._projector(inputs)
        embeddings, mask = self._encoder(embeddings, mask)
        return embeddings
