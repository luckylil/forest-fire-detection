# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3,C2f_dysnakeconv,SPPF_LSKA,C3_DAttention, C2f_DAttention,
                    C2f_Faster,C3_Faster,C2f_CloAtt,C3_CloAtt)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention,C2f_triple,RCSOSA,RepVGG,LSKA,DAttention,
                  SwinTransformer,EfficientAttention)
from .head import (Classify, Detect, Pose, RTDETRDecoder, Segment,DetectAux,
                   Detect_AFPN_P2345, Detect_AFPN_P2345_Custom, Detect_AFPN_P345, Detect_AFPN_P345_Custom)
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .AFPN import ASFF_2, ASFF_3, ASFF_4
from .attention import TripletAttention
from .kernel_warehouse import KWConv, Warehouse_Manager
from .VanillaNet import VanillaBlock

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP','C2f_triple','RepVGG','LSKA',
           'SwinTransformer','RCSOSA','SPPF_LSKA','Detect_AFPN_P2345', 'Detect_AFPN_P2345_Custom',
           'Detect_AFPN_P345', 'Detect_AFPN_P345_Custom','C3_DAttention', 'C2f_DAttention','DAttention','ASFF_2','ASFF_3','ASFF_4',
           'C2f_Faster','C3_Faster','C2f_CloAtt','C3_CloAtt','EfficientAttention','TripletAttention','KWConv', 'Warehouse_Manager','DetectAux','VanillaBlock')


