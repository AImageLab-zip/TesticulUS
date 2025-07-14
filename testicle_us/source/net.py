
import torch, torchvision, sys
import torch.nn as nn
from torchvision import models
from pathlib import Path
from hico.net import *
from pathlib import Path
# def fix_conv1(net: nn.Module):
#     net[0] = nn.Conv2d(
#         in_channels=1,
#         out_channels=net[0].out_channels,
#         kernel_size=net[0].kernel_size,
#         stride=net[0].stride,
#         padding=net[0].padding,
#         bias=net[0].bias is not None
#     )
#     nn.init.kaiming_normal_(net[0].weight)
#     return net


def fix_conv1(net: nn.Module):
    original_conv1 = net[0]
    original_weight = original_conv1.weight.data
    new_weight = torch.mean(original_weight, dim=1, keepdim=True)
    net[0] = nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                       stride=original_conv1.stride, padding=original_conv1.padding, bias=False)
    net[0].weight.data = new_weight
    return net


def load_uscl(weights_path: str):
    uscl_weights = torch.load(weights_path, weights_only=True)
    state_dict = {}
    for k in uscl_weights.keys():
        if k.startswith('l') | k.startswith('fc'):
            continue
        new_k = k.removeprefix("features.")
        state_dict[new_k] = uscl_weights[k]
    backbone = nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())[:-1])
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(msg)
    print(f"Loaded uscl weigts")
    return backbone


class BackboneWrapper(nn.Module):
    def __init__(self, out_dim=2048, num_classes=2, backbone: str = None, weights_path: str = None):
        super(BackboneWrapper, self).__init__()
        self.backbone = backbone
        self.weights_path = weights_path
        if backbone == "rn18":
            if weights_path != "scratch":
                self.backbone = nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())[:-1])
                print("Using resnet18 imagenet weights")

            else:
                self.backbone = nn.Sequential(*list(models.resnet18().children())[:-1])
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)  # He initialization
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                print("Using resnet18 scratch weights")

            self.embed_dim = self.backbone[-2][1].bn2.num_features
        elif backbone == "uscl":
            assert Path(weights_path).is_file(), "invalid uscl weights path!!"
            self.backbone = load_uscl(weights_path)
            self.embed_dim = self.backbone[-2][1].bn2.num_features
            print("Using uscl")
        elif backbone == "rn50":
            if weights_path != "scratch":
                self.backbone = nn.Sequential(*list(models.resnet50(weights='DEFAULT').children())
                                              [:-1])
                print("Using resnet50 imagenet weights")
            else:
                self.backbone = nn.Sequential(*list(models.resnet50().children())
                                              [:-1])
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)  # He initialization
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                print("Using resnet50 scratch weights")

            self.embed_dim = self.backbone[-2][2].bn3.num_features
        elif backbone == "resnet_fpn":
            if weights_path != "scratch":
                self.backbone = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=True)
                print("Using resnet_fpn with pretrained pytorch weights")
            else:
                self.backbone = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=False)
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)  # He initialization
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                print("Using resnet_fpn with scrathc weights")

            self.embed_dim = 256
        elif backbone == None:
            print("Backbone: None")
            sys.exit(-1)
        elif backbone == "densenet":
            if weights_path == "scratch" or weights_path == None:
                self.backbone = models.densenet121()
                self.backbone.classifier = torch.nn.Identity()
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                print("Using densenet scratch- weights")
            elif Path(weights_path).is_file():
                self.backbone = models.densenet121(weights="DEFAULT")
                self.backbone.classifier = torch.nn.Identity()
                # checkpoint = torch.load(weights_path, weights_only=False)
                # msg = self.backbone.load_state_dict(checkpoint["model_state_dict"])
                print(msg)
                print(f"Using dense using weights at {weights_path}")
            else:
                self.backbone = models.densenet121(weights="DEFAULT")
                self.backbone.classifier = torch.nn.Identity()
                print("Using densenet imagenet weights")

            self.embed_dim = 1024

        self.out_dim = out_dim
        self.num_classes = num_classes
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.out_dim)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x).squeeze()
        return self.contrastive_head(features), self.cls_head(features)

    @classmethod
    def create_backbone(cls, backbone, out_dim=2048, num_classes=2, weights_path=None):
        """Factory method to create the appropriate backbone"""
        if backbone == "resnet_fpn_hico":
            print("Using rn_fpn for HiCo pretrain")
            return ResNetSimCLR()
        else:
            return cls(out_dim, num_classes, backbone, weights_path)


class ClassificationWrapper(nn.Module):
    def __init__(self, backbone=None, num_classes=2, weights_path=None):
        super(ClassificationWrapper, self).__init__()
        if backbone == "rn18":
            if weights_path == "scratch" or weights_path == None:
                self.backbone = nn.Sequential(*list(models.resnet18().children())
                                              [:-1])
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                print("Using rn18 scratch weights")
            elif Path(weights_path).is_file():
                self.backbone = nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())
                                              [:-1])
                checkpoint = torch.load(weights_path, weights_only=False)
                msg = self.backbone.load_state_dict(checkpoint["model_state_dict"])
                print(msg)
                print(f"Using rn18 using weights at {weights_path}")
            else:
                self.backbone = nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())
                                              [:-1])
                print("Using rn18 imagenet weights")
            self.embed_dim = self.backbone[-2][1].bn2.num_features
        elif backbone == "uscl":
            self.backbone = load_uscl(weights_path)
            self.embed_dim = self.backbone[-2][1].bn2.num_features
            print("Using uscl as model")
        elif backbone == "hico":
            self.backbone = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=True)
            state_dict = torch.load(weights_path)
            new_dict = {k: state_dict[k] for k in list(state_dict.keys())
                        if not (k.startswith('l')
                                | k.startswith('fc'))}  # # discard MLP and fc
            model_dict = self.backbone.state_dict()
            model_dict.update(new_dict)
            self.backbone.load_state_dict(model_dict)
            self.embed_dim = 256

            print('\nThe self-supervised trained parameters are loaded.\n')
        elif backbone == "resnet_fpn":
            if weights_path == None or weights_path == "scratch":
                self.backbone = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=False)
                print("Using resnet_fpn with scratch weights")
            elif Path(weights_path).is_file():
                self.backbone = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=True)
                state_dict = torch.load(
                    weights_path, weights_only=False)["model_state_dict"]
                new_dict = {"features." + str(k): state_dict[k] for k in list(state_dict.keys())}
                # if not (k.startswith('l')
                # | k.startswith('fc'))}  # # discard MLP and fc
                model_dict = self.backbone.state_dict()
                model_dict.update(new_dict)
                msg = self.backbone.load_state_dict(model_dict, strict=False)
                print(msg)
                print(f"Using resnet_fpn with pretrained weights at {weights_path}")
            else:
                self.backbone = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=True)
                print("Using resnet_fpn with pretrained pytorch weights")

            self.embed_dim = 256
        elif backbone == "rn50":
            if weights_path == None or weights_path == "scratch":
                self.backbone = nn.Sequential(*list(models.resnet50().children())
                                              [:-1])
                print("Using rn50 with scratch weights")
            else:
                self.backbone = nn.Sequential(*list(models.resnet50(weights='DEFAULT').children())
                                              [:-1])
                print("Using rn50 with pretrained pytorch weights")

            self.embed_dim = self.backbone[-2][2].bn3.num_features
        elif backbone == "densenet":
            if weights_path == "scratch" or weights_path == None:
                self.backbone = models.densenet121()
                self.backbone.classifier = torch.nn.Identity()
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                print("Using densenet scratch- weights")
            elif Path(weights_path).is_file():
                self.backbone = models.densenet121(weights="DEFAULT")
                self.backbone.classifier = torch.nn.Identity()
                # checkpoint = torch.load(weights_path, weights_only=False)
                # msg = self.backbone.load_state_dict(checkpoint["model_state_dict"])
                print(msg)
                print(f"Using dense using weights at {weights_path}")
            else:
                self.backbone = models.densenet121(weights="DEFAULT")
                self.backbone.classifier = torch.nn.Identity()
                print("Using densenet imagenet weights")

            self.embed_dim = 1024

        self.num_classes = num_classes

        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_classes),
        )

        self.initialize_weights_()

    def initialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_backbone(self, flag=True):
        for param in self.backbone.parameters():
            param.requires_grad = not flag

    def unfreeeze_backbone(self):
        self.freeze_backbone(False)

    def forward(self, x):
        features = self.backbone(x).squeeze()
        return self.cls_head(features)

    def get_embedding(self, x):
        return self.backbone(x).squeeze()
