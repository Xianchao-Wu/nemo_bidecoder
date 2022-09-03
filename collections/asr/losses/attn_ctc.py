import torch
import torch.nn.functional as F
from typeguard import check_argument_types
from nemo.collections.asr.utils.common import BLANK_ID


class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        normalize_length: bool = True
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs (vocabulary size, with <blank>, <unk>, <sos/eos>)
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
            normalize_length: true=use token.num for loss normalization; false=use batch.size
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(blank=BLANK_ID, reduction=reduction_type, zero_infinity=True) 
        # NOTE important!!!
        # blank=blank.id TODO 0 or 3605 or other values? -> =0 is better (to fit inference algorithms)
        # better to be, <blank>=0, <unk>=1, ..., <sos/eos>=3605+3-1=3607
        # 增加了三个tokens, <blank>=0, <unk>=1, <sos/eos>=3607

        # 目前已有的是，3605个characters: id=2 to 3605+2-1=3606

        self.normalize_length = normalize_length

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        #import ipdb; ipdb.set_trace()
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        denom = torch.sum(ys_lens).item() if self.normalize_length else ys_hat.size(1)
        loss = loss / denom #ys_hat.size(1)
        return loss

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
