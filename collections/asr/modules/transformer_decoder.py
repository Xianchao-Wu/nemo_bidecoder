# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Decoder definition."""
from typing import Tuple, List, Optional

import torch
from typeguard import check_argument_types

#from wenet.transformer.attention import MultiHeadedAttention
from nemo.collections.asr.modules.attention import MultiHeadedAttention

#from wenet.transformer.decoder_layer import DecoderLayer
from nemo.collections.asr.modules.transformer_decoder_layer import DecoderLayer

#from wenet.transformer.embedding import PositionalEncoding
from nemo.collections.asr.modules.transformer_embedding import PositionalEncoding

#from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from nemo.collections.asr.modules.positionwise_feed_forward import PositionwiseFeedForward

#from wenet.utils.mask import (subsequent_mask, make_pad_mask)
#from nemo.collections.asr.modules.mask import (subsequent_mask, make_pad_mask)
#from nemo.collections.common.mask import (subsequent_mask, make_pad_mask)
from nemo.collections.asr.utils.mask import (subsequent_mask, make_pad_mask)

from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule

from omegaconf.listconfig import ListConfig

__all__ = ['TransformerDecoder', 'BiTransformerDecoder']


#class TransformerDecoder(torch.nn.Module):
class TransformerDecoder(NeuralModule, Exportable):
    """Base class of Transfomer decoder module.
    Args:
        num_classes (vocab_size): output dim
        feat_in: (encoder_output_size): dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        num_classes: int, #vocab_size: int,
        feat_in: int, #encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = feat_in # encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(num_classes + 1, #vocab_size, TODO num_classes or num_classes+1 (for sos/eos)? 
                    attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, num_classes) #vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    num_classes=vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache


#class BiTransformerDecoder(torch.nn.Module):
class BiTransformerDecoder(NeuralModule, Exportable):
    """Base class of Transfomer decoder module.
    Args:
        num_classes (vocab_size): output dim
        feat_in (encoder_output_size): dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        num_classes: int, #vocab_size+1: int,
        feat_in: int, #encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
        #vocabulary: list = None,
        vocabulary: ListConfig = None,
    ):

        assert check_argument_types()
        super().__init__()
        
        if vocabulary is not None:
            if num_classes != len(vocabulary) + 1: # TODO 
                raise ValueError(
                    f"If vocabulary is specified, it's length+1 should = num_classes. "
                    f"Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        self._num_classes = num_classes #3606 already, don't + 1. id=3605 for eos=sos  

        self.left_decoder = TransformerDecoder(
            num_classes, #vocab_size, 
            feat_in, #encoder_output_size, 
            attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after)

        self.right_decoder = TransformerDecoder(
            num_classes, #vocab_size, 
            feat_in, #encoder_output_size, 
            attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    num_classes=vocab_size+1) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, num_classes=vocab_size+1)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad,
                                               ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)
    
    def input_example(self, max_batch=1, max_dim=256):
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def vocabulary(self):
        return self.__vocabulary
    
    @property
    def num_classes_with_blank(self):
        return self._num_classes # 3606; id=3605 is for sos/eos

