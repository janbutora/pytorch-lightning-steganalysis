import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
from efficientnet_pytorch import EfficientNet
from torch import nn
import timm
import types

from SRNet import SRNet

zoo_params = {

    'srnet': {
        'fc_name': 'fc',
        'fc': nn.Linear(in_features=512, out_features=2, bias=True),
        'init_op': partial(SRNet, in_channels=3, nclasses=2)
    },

    'efficientnet-b0': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1280, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b0')
    },

    'efficientnet-b2': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1408, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b2')
    },

    'efficientnet-b4': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1792, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b4')
    },

    'efficientnet-b5': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2048, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b5')
    },

    'efficientnet-b6': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2304, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b6')
    },

    'efficientnet-b7': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2560, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b7')
    },

    'mixnet_xl': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'mixnet_xl', pretrained=True)
    },

    'mixnet_s': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=True)
    },


    'seresnet18': {
        'fc_name': 'last_linear',
        'fc': nn.Linear(in_features=512, out_features=4, bias=True),
        'init_op':  partial(timm.create_model, 'seresnet18', pretrained=True)
    },
}

def get_net(model_name):
    net = zoo_params[model_name]['init_op']()
    setattr(net, zoo_params[model_name]['fc_name'], zoo_params[model_name]['fc'])
    return net

def to_grayscale_srnet(net):
    net.prelayer = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=True)
    def new_forward_features(self, inputs):
        # Stem
        x = self.prelayer(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pooling(x)
        return x

    net.forward_features = types.MethodType(new_forward_features, net)

    return net
