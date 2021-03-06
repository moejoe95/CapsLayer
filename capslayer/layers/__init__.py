# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from capslayer.layers.convolutional import conv1d
from capslayer.layers.convolutional import conv2d
from capslayer.layers.convolutional import conv3d
from capslayer.layers.layers import primaryCaps
from capslayer.layers.attention import selfAttention
from capslayer.layers.layers import dense
from capslayer.layers.layers import dense as fully_connected
from capslayer.layers.residual_layers import residualConvs
from capslayer.layers.residual_layers import residualCapsNetwork
from capslayer.layers.residual_layers import getParamsSkip
from capslayer.layers.dropout import dropout

__all__ = [
        'conv1d', 
        'conv2d', 
        'conv3d', 
        'selfAttention', 
        'primaryCaps', 
        'dense', 
        'fully_connected', 
        'residualConvs', 
        'getParamsSkip',
        'residualCapsNetwork',
        'dropout'
        ]
