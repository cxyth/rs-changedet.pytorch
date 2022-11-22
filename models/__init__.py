# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 构建模型的入口
'''
from .unet import Unet
# from .unetplusplus import UnetPlusPlus
from .fpn import FPN
from .linknet import Linknet
# from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus


def create_model(cfg: dict):
    model_type = cfg['type']
    arch = cfg['arch']
    if model_type == 'smp':
        import segmentation_models_pytorch as smp
        smp_net = getattr(smp, arch)

        encoder = cfg.get('encoder', 'resnet34')
        pretrained = cfg.get('pretrained', 'imagenet')
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        aux_params = cfg.get('aux_params', None)

        model = smp_net(               # smp.UnetPlusPlus
            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=pretrained,     # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channel,     # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channel,     # model output channels (number of classes in your dataset)
            aux_params=aux_params
        )
    elif model_type == 'siamese':
        # 自定义模型
        archs = [Unet, FPN, Linknet, DeepLabV3, DeepLabV3Plus]
        archs_dict = {a.__name__.lower(): a for a in archs}
        try:
            model_class = archs_dict[arch.lower()]
        except KeyError:
            raise KeyError("Wrong architecture type `{}`. Available options are: {}".format(
                arch, list(archs_dict.keys()),
            ))
        encoder = cfg.get('encoder', 'resnet34')
        pretrained = cfg.get('pretrained', 'imagenet')
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        aux_params = cfg.get('aux_params', None)
        return model_class(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=in_channel,
            classes=out_channel,
            aux_params=aux_params,
        )
    else:
        print('type error')
        exit()
    return model


