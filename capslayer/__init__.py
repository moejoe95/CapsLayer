from __future__ import absolute_import

from capslayer import layers
from capslayer import decoders
from capslayer import datasets
from capslayer.ops import losses
from capslayer import summary
from capslayer.ops.losses import losses

from capslayer.ops.nn_ops import space_to_batch_nd_v1 as space_to_batch_nd
from capslayer.ops.nn_ops import batch_to_space_nd
from capslayer.ops.nn_ops import softmax

from capslayer.ops.math_ops import matmul_v2 as matmul
from capslayer.ops.math_ops import norm
from capslayer.ops.math_ops import reduce_sum
from capslayer.ops.math_ops import divide
from capslayer.ops.math_ops import log

from capslayer.ops.ops import shape

_allowed_symbols = [
    'layers',
    'decoders',
    'datasets',
    'distributions',
    'losses',
    'summary',
    'space_to_batch_nd',
    'batch_to_space_nd',
    'softmax',
    'matmul',
    'norm',
    'divide',
    'log',
    'shape'
]

__version__ = "0.1.5"

__all__ = [s for s in _allowed_symbols]
