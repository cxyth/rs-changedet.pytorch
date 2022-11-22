import torch
import torch.nn.functional as F
from . import initialization as init


# class SegmentationModel(torch.nn.Module):
#
#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def forward(self, x):
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""
#         features = self.encoder(x)
#         decoder_output = self.decoder(*features)
#
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks
#
#     def predict(self, x):
#         """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
#
#         Args:
#             x: 4D torch tensor with shape (batch_size, channels, height, width)
#
#         Return:
#             prediction: 4D torch tensor with shape (batch_size, classes, height, width)
#
#         """
#         if self.training:
#             self.eval()
#
#         with torch.no_grad():
#             x = self.forward(x)
#
#         return x


class ChangeDetModel(torch.nn.Module):

    def initialize(self):
        # init.initialize_decoder(self.changedetect_decoder)
        # init.initialize_head(self.changedetect_head)
        # init.initialize_decoder(self.segmentation_decoder)
        # init.initialize_head(self.segmentation_head)
        init.initialize_decoder(self.changedetect_decoder)
        init.initialize_head(self.changedetect_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x1, x2):

        features1 = self.encoder(x1)
        features2 = self.encoder(x2)

        # out1 = self.segmentation_decoder(*features1)
        # out1 = self.segmentation_head(out1)
        # out2 = self.segmentation_decoder(*features2)
        # out2 = self.segmentation_head(out2)

        cats = [torch.cat([c1, c2], dim=1) for c1, c2 in zip(features1, features2)]
        proj = self.changedetect_project(*cats)
        cd = self.changedetect_decoder(*proj)
        out_cd = self.changedetect_head(cd)

        return out_cd
