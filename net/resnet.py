from torchvision import models
from .common import *


resnet_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152
}


class ResNetFc(nn.Module):
    def __init__(self, resnet_name, bottleneck_size=256, use_bottleneck=True, num_classes=1000, plug_position=7):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_size) if use_bottleneck else Identity()
        self.bottleneck.apply(init_weights)

        self.num_classes = num_classes

        if use_bottleneck:
            self.fc = nn.Linear(bottleneck_size, num_classes)
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
        self.fc.apply(init_weights)

        self.layers = PluggableSequential(
            # 0
            nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.maxpool
            ),
            # 1
            self.layer1,
            # 2
            self.layer2,
            # 3
            self.layer3,
            # 4
            self.layer4,  # <- Transformer!
            # 5
            nn.Sequential(
                self.avgpool,
                nn.Flatten()
            ),
            # 6
            self.bottleneck,  # Might be Identity if bottleneck is disabled
            # 7
            self.fc,
            # 8
            plug_position=plug_position
        )

        self.plug_position = plug_position
        self.__in_features = bottleneck_size

    def forward(self, x):
        x, features = self.layers(x)

        return x, features

    def output_size(self):
        # TODO change if changing plug_position!!!
        assert self.plug_position == 7
        return self.fc.in_features

    def get_parameters(self, base_lr, base_wd):
        parameter_list = [
            {
                "params": itertools.chain(
                    self.conv1.parameters(),
                    self.bn1.parameters(),
                    self.maxpool.parameters(),
                    self.layer1.parameters(),
                    self.layer2.parameters(),
                    self.layer3.parameters(),
                    self.layer4.parameters(),
                    self.avgpool.parameters()
                ),
                "lr": base_lr,
                'weight_decay': base_wd * 2
            },
            {
                "params": self.bottleneck.parameters(),
                "lr": base_lr * 10,
                'weight_decay': base_wd * 2
            },
            {
                "params": self.fc.parameters(),
                "lr": base_lr * 10,
                'weight_decay': base_wd * 2
            }
        ]

        return parameter_list

    def freeze(self) -> FrozenContext:
        return FrozenContext(self.parameters())


def generate_fc_stack(in_features: int, out_features: int, num_fc_layers: int, num_units: int,
                      dropout: float = 0) -> nn.Module:
    """
    Generate a variable-size fully connected block
    :param in_features:
    :param out_features:
    :param num_fc_layers:
    :param num_units:
    :param dropout:
    :return:
    """
    assert num_fc_layers >= 1

    # List of layers
    fc_layers = []
    # -1 because the final layer will be added separately
    for i in range(num_fc_layers - 1):
        # If this is the first layer, its input size is bound to match in_features
        fc_in_features = in_features if i == 0 else num_units
        # Output size is fixed (for now)
        fc_out_features = num_units
        # Add the layer with its activation (ReLU) and Dropout if required
        modules = [
            nn.Linear(in_features=fc_in_features, out_features=fc_out_features),
            nn.ReLU()
        ]
        if dropout != 0.:
            modules.append(nn.Dropout(p=dropout))
        fc_layers.append(nn.Sequential(*modules))

    # Add the final layer
    fc_layers.append(
        # in_features if there are not previous layers (this is the only one)
        nn.Linear(in_features=in_features if len(fc_layers) == 0 else num_units, out_features=out_features)
    )

    return nn.Sequential(*fc_layers)


class Discriminator(nn.Module):
    def __init__(self, in_feature, num_classes, hidden_size=1024, num_fc_layers=3, dropout=0.5):
        super(Discriminator, self).__init__()
        self.layers = generate_fc_stack(in_features=in_feature,
                                        out_features=num_classes,
                                        num_fc_layers=num_fc_layers,
                                        num_units=hidden_size,
                                        dropout=dropout)

        self.layers.apply(init_weights)
        self.grl_module = GradientReversalLayer()

    def forward(self, x, lambda_v):
        x = self.grl_module(x, lambda_v)
        y = self.layers(x)
        return y

    def get_parameters(self, base_lr, base_wd):
        return [{
            "params": self.layers.parameters(),
            "lr": base_lr * 10,
            'weight_decay': base_wd * 2
        }]
