import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNetNew(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        f = self.avgpool(c5)

        return c2, c3, c4, c5, f


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetNew(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[arch]))
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class FPNresnet18(nn.Module):
    def __init__(self, pretrained=False):
        super(FPNresnet18, self).__init__()

        self.R18 = resnet18(pretrained)
        # self.features = nn.Sequential(*list(R18.children())[:-1])

        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        # return nn.functional.upsample(x, size=(H, W), mode='bilinear') + y
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c2, c3, c4, c5, f = self.R18(x)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5, f


class ResNetUSCL(nn.Module):
    ''' The ResNet feature extractor + projection head + classifier for USCL '''

    def __init__(self, base_model, out_dim, pretrained=False, input_dim=256):
        super(ResNetUSCL, self).__init__()
        self.input_dim = input_dim
        self.features = FPNresnet18(True)       # default=True
        # self.features = FPNresnet18(False)      # scratch
        self.l2 = nn.Linear(256 * self.input_dim // 4 * self.input_dim // 4, out_dim)
        self.l3 = nn.Linear(256 * self.input_dim // 8 * self.input_dim // 8, out_dim)
        self.l4 = nn.Linear(256 * self.input_dim // 16 * self.input_dim // 16, out_dim)
        self.l5 = nn.Linear(256 * self.input_dim // 32 * self.input_dim // 32, out_dim)
        # resnet50fpn 更改 512 为 2048
        self.lf = nn.Linear(512 * 1 * 1, out_dim)

        # projection MLP
        # self.linear = nn.Linear(1*out_dim, out_dim)
        self.linear = nn.Linear(5 * out_dim, out_dim)

        # classifier
        num_classes = 3
        self.fc = nn.Linear(out_dim, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        p2, p3, p4, p5, h = self.features(x)
        f2, f3, f4, f5, fp = \
            torch.flatten(p2, start_dim=1), torch.flatten(p3, start_dim=1), torch.flatten(
                p4, start_dim=1), torch.flatten(p5, start_dim=1), torch.flatten(h, start_dim=1)
        x2 = F.relu(self.l2(f2))
        x3 = F.relu(self.l3(f3))
        x4 = F.relu(self.l4(f4))
        x5 = F.relu(self.l5(f5))
        xf = F.relu(self.lf(fp))

        # x = xf
        # x = self.linear(x)
        x = torch.cat((x2, x3, x4, x5, xf), 1)
        x = self.linear(x)

        return x


class ResNetSimCLR(nn.Module):
    ''' The ResNet feature extractor + projection head for SimCLR '''

    def __init__(self, out_dim=256, pretrained=True, input_dim=256):
        super(ResNetSimCLR, self).__init__()
        self.input_dim = input_dim
        # projection MLP
        self.features = FPNresnet18(pretrained)
        self.backbone = self.features
        # self.features = FPNresnet18(False)         # default = True
        self.l2 = nn.Linear(256 * self.input_dim // 4 * self.input_dim // 4, out_dim)
        self.l3 = nn.Linear(256 * self.input_dim // 8 * self.input_dim // 8, out_dim)
        self.l4 = nn.Linear(256 * self.input_dim // 16 * self.input_dim // 16, out_dim)
        self.l5 = nn.Linear(256 * self.input_dim // 32 * self.input_dim // 32, out_dim)

        # resnet50fpn 更改 512 为 2048
        self.lf = nn.Linear(512 * 1 * 1, out_dim)  # out_dim = 256

        self.l22 = nn.Linear(out_dim, out_dim)
        self.l33 = nn.Linear(out_dim, out_dim)
        self.l55 = nn.Linear(out_dim, out_dim)

        #########################################################
        num_classes = 2
        self.fc = nn.Linear(1 * out_dim, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        p2, p3, p4, p5, h = self.features(x)
        h1 = h.squeeze()  # feature before project g()=h1

        f2, f3, f4, f5, fp = \
            torch.flatten(p2, start_dim=1), torch.flatten(p3, start_dim=1), torch.flatten(
                p4, start_dim=1), torch.flatten(p5, start_dim=1), torch.flatten(h, start_dim=1)
        x2 = F.relu(self.l2(f2))
        x3 = F.relu(self.l3(f3))
        x4 = F.relu(self.l4(f4))
        x5 = F.relu(self.l5(f5))
        xf = F.relu(self.lf(fp))

        # Version 1
        c = xf
        c = self.fc(c)

        xx2 = F.relu(self.l2(f2))    # From p2: l2 -> relu -> l22
        xx2 = self.l22(xx2)

        xx3 = F.relu(self.l3(f3))    # From p3: l3 -> relu -> l33
        xx3 = self.l33(xx3)

        xx5 = F.relu(self.l5(f5))    # From p5: l5 -> relu -> l55
        xx5 = self.l55(xx5)

        return xx2, xx3, xx5, c
