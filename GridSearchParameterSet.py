from torch import nn as nn


class GridSearchParameterSet:

    def compare(self, other: 'GridSearchParameterSet') -> bool:
        return (
            self.nLayers == other.nLayers and
            self.flatten == other.flatten and
            self.downSample == other.downSample and
            self.leaky_cnn == other.leaky_cnn and
            self.leaky_classifier == other.leaky_classifier and
            self.base_channels == other.base_channels and
            self.kernel_size == other.kernel_size and
            self.padding == other.padding and
            self.output_height == other.output_height and
            self.output_width == other.output_width and
            self.classifier_layers == other.classifier_layers and
            self.classifier_hidden == other.classifier_hidden and
            self.dropout == other.dropout
        )

    @staticmethod
    def fromDict(d):
        return GridSearchParameterSet(
            nLayers=d.get('nlayers', 4),
            flatten=d.get('flatten', False),
            downSample=d.get('downSample', False),
            leaky_cnn=d.get('leaky_cnn', False),
            leaky_classifier=d.get('leaky_classifier', False),
            base_channels=d.get('base_channels', 16),
            kernel_size=d.get('kernel_size', 3),
            padding=d.get('padding', 0),
            classifier_layers=d.get('classifier_layers', 2),
            classifier_hidden=d.get('classifier_hidden', 128),
            dropout=d.get('dropout', 0.0)
        )

    def __init__(
            self,
            nLayers: int = 4,
            flatten: bool = False,
            downSample:bool = False,
            leaky_cnn: bool = False,
            leaky_classifier = False,
            base_channels = 16,
            kernel_size =3,
            padding = 0,
            output_height=4,
            output_width = 4,
            classifier_layers=2,  # Number of classifier layers
            classifier_hidden=128, # Hidden size for classifier layers,
            dropout=0.0

        ):
        self.feature_maps = []
        n_output_channels = -1
        for layer_index in range(0,nLayers):
            previous_base_channels = 1
            if layer_index != 0:
                previous_base_channels = base_channels
                base_channels *= 2
                n_output_channels = base_channels

            self.feature_maps.append(nn.Conv2d(previous_base_channels, base_channels, kernel_size=kernel_size, padding=padding))

            if leaky_cnn:
                self.feature_maps.append(nn.LeakyReLU())
            else:
                self.feature_maps.append(nn.ReLU())

            if layer_index == nLayers -1:
                self.feature_maps.append(nn.AdaptiveAvgPool2d((output_height, output_width)))

            elif downSample is not None and layer_index % 2 == 1:
                self.feature_maps.append(nn.MaxPool2d(downSample))

        self.cnn = nn.Sequential(*self.feature_maps)

        if leaky_classifier:
            rel = nn.LeakyReLU()
        else:
            rel = nn.ReLU()


        self.classifier_params = []
        input_dim = n_output_channels * output_height * output_width
        for _ in range(classifier_layers - 1):
            self.classifier_params.append(nn.Linear(input_dim, classifier_hidden))
            self.classifier_params.append(rel)
            input_dim = classifier_hidden
        self.classifier_params.append(nn.Linear(input_dim, 1))  # Final output layer
        self.classifier = nn.Sequential(*self.classifier_params)
