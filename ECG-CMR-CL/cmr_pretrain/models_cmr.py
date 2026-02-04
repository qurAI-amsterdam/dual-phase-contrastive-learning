import torch
from torch import nn
from torchvision.models import resnet50
from abc import ABC, abstractmethod


class CMRModel(torch.nn.Module, ABC):
    def __init__(self, input_size:int=2, return_latent_space:bool=False,
                 pretrained:bool=False, contrastive_learning:bool=False,
                 latent_dim:int=200, num_outputs:int=1, args=None):
        super().__init__()
        # Explicitly convert to Python native types
        self.input_size = int(input_size)
        self.return_latent_space = bool(return_latent_space)
        self.pretrained = bool(pretrained)
        self.contrastive_learning = bool(contrastive_learning)
        self.args = args

        # Ensure these are proper integers
        if latent_dim is None:
            self.latent_dim = 200
        else:
            try:
                self.latent_dim = int(latent_dim)
            except (ValueError, TypeError):
                print(f"Warning: Invalid latent_dim value '{latent_dim}', using default 200")
                self.latent_dim = 200

        if num_outputs is None:
            self.num_outputs = 1
        else:
            try:
                self.num_outputs = int(num_outputs)
            except (ValueError, TypeError):
                print(f"Warning: Invalid num_outputs value '{num_outputs}', using default 1")
                self.num_outputs = 1

        # Print debug info
        print(f"Initializing CMRModel with latent_dim={self.latent_dim}, num_outputs={self.num_outputs}")

        self.model = self.initialize_model(pretrained=self.pretrained)

        # Adaptive Pooling for variable input sizes
        if self.args.model_name == "ResNet50-4D":
            self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Adaptive Pooling for variable input sizes
        elif self.args.model_name == "ResNet50-3D-MLP":
            self.global_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Get the correct feature dimension from the backbone
        self.feature_dim = self._get_feature_dim()

        # Fully Connected layer (latent space) - corrected this to use latent_dim
        if hasattr(self, 'lstm'):
            pass
        else:    
            self.fc = nn.Linear(self.feature_dim, self.latent_dim)

        # Output layer for regression
        self.output_layer = nn.Linear(self.latent_dim, self.num_outputs)

    def forward(self, x):

        # print("self:", self)
        # exit()

        if self.args.model_name in ["ResNet50", "ResNet50-3D-MLP"]:
            x = self.model(x)
        else:
            encoded_frames = []
            timepoints = x.shape[2]

            # encode individually the slices for each time point
            for i in range(timepoints):
                frame = x[:, :, i] # Shape: (batch_size, channels, height, width)

                encoded_frame = self.model(frame) # Encode each frame
                encoded_frames.append(encoded_frame)

            x = torch.stack(encoded_frames, dim=1)

        x = self.global_avg_pool(x)

        if hasattr(self, 'lstm'):
            x = torch.flatten(x, 2)

            x = self.fc(x)
            x, _ = self.lstm(x)
            # select the last step from the lstm
            x = x[:,-1]

        else:
            if self.args.model_name == "ResNet50-3D-MLP":
                x = x.squeeze(-1).squeeze(-1)
                x = self.temporal_mlp(x)

            x = torch.flatten(x, 1)

            # Latent space representation
            x = self.fc(x)

        return x if self.return_latent_space else self.output_layer(x)

    # Abstract methods that are implemented in subclasses
    @abstractmethod
    def initialize_model(self, pretrained: bool):
        """It is not possible to initialize the base model.
            Must be implemented in the model's subclass."""
        pass

    @abstractmethod
    def _get_feature_dim(self):
        """Helper method to determine the feature dimension from the backbone"""
        pass


class ResNet50(CMRModel):
    def initialize_model(self, pretrained: bool):
        # Get ResNet50 model
        if pretrained:
            model = resnet50(weights='IMAGENET1K_V1') # weights='IMAGENET1K_V1'
        else:
            model = resnet50()

        # Modify first layer to handle different input channels
        model.conv1 = nn.Conv2d(self.input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove avg pool and fc to get the CMR encoder
        self.encoder = nn.Sequential(*list(model.children())[:-2]) 
        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in model.parameters():
                param.requires_grad = False
            return model

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return self.encoder

    def _get_feature_dim(self):
        """Return the feature dimension for ResNet50"""
        return 2048  # ResNet50 always outputs 2048 features


class ResNet503D(CMRModel):
    def initialize_model(self, pretrained: bool):
        # Get ResNet50 model for 3D data          
        if pretrained:
            model = resnet50(weights='IMAGENET1K_V1') # weights='IMAGENET1K_V1'
        else:
            model = resnet50()

        # Modify first layer to handle different input channels
        model.conv1 = nn.Conv2d(self.input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove avg pool and fc to get the CMR encoder
        self.encoder = nn.Sequential(*list(model.children())[:-2])

        self.lstm_layers = 1

        # Fully Connected layer (latent space) - corrected this to use latent_dim
        self.fc = nn.Linear(self._get_feature_dim(), self.latent_dim)

        print("self.latent_dim shape:", self.latent_dim)

        # LSTM on top of the encoder to process the temporal dimension of the data
        self.lstm = nn.LSTM(input_size=self.latent_dim, #//self.timepoints * 2,
                    hidden_size=self.latent_dim,
                    num_layers=self.lstm_layers, batch_first=True)

        print(self)

        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in model.parameters():
                param.requires_grad = False
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return self.encoder

    def _get_feature_dim(self):
        """Return the feature dimension for ResNet50"""
        return 2048  # ResNet50 always outputs 2048 features

class ResNet503D_MLP(CMRModel):
    def initialize_model(self, pretrained: bool):
        # Get ResNet50 model            
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)

        # Modify first layer to handle different input channels
        print("self.input_size", self.input_size)

        # Modify the first convolution layer to match number of channels, grayscale images (MRI) 
        first_conv_layer = model.blocks[0].conv
        model.blocks[0].conv = nn.Conv3d(
            self.input_size, 
            first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=False
        )

        # Remove last block, to return latent space
        self.encoder = nn.Sequential(*model.blocks[:-1])

        print("self.latent_dim:", self.latent_dim)
        print("self.args.temporal_dim:", self.args.temporal_dim)

        self.temporal_mlp = nn.Linear(self.args.temporal_dim, 1)

        print(model)

        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.temporal_mlp.parameters():
                param.requires_grad = False

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return self.encoder

    def _get_feature_dim(self):
        """Return the feature dimension for ResNet50"""
        return 2048  # ResNet50 always outputs 2048 features


class ResNet504D(CMRModel):
    def initialize_model(self, pretrained: bool):
        # Get ResNet50 model            
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)

        # Modify first layer to handle different input channels
        print("self.input_size", self.input_size)

        # Modify the first convolution layer to match number of channels, grayscale images (MRI) 
        first_conv_layer = model.blocks[0].conv
        model.blocks[0].conv = nn.Conv3d(
            self.input_size, 
            first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=False
        )

        # Remove last block, to return latent space
        self.encoder = nn.Sequential(*model.blocks[:-1])

        print(model)

        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in model.parameters():
                param.requires_grad = False

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return self.encoder

    def _get_feature_dim(self):
        """Return the feature dimension for ResNet50"""
        return 2048  # ResNet50 always outputs 2048 features


class SwinTransformer(CMRModel):
    def initialize_model(self, pretrained:bool):
        # TODO: Implement
        raise NotImplementedError("SwinTransformer implementation is incomplete")

    def _get_feature_dim(self):
        """Override to provide the correct feature dimension for Swin model"""
        return 768  # Example for Swin-T


def get_model(model_name, args):
    # Debug the input args
    print(f"Creating model: {model_name}")

    print(f"Trying args: input_size={args.input_size}, latent_dim={args.latent_dim}, num_outputs={args.num_outputs}")

    try:
        # Validate that critical args are not None and are of expected types
        input_size = int(args.input_size)
        latent_dim = int(args.latent_dim)
        num_outputs = int(args.num_outputs)

        print(f"Args: input_size={input_size}, latent_dim={latent_dim}, num_outputs={num_outputs}")

        if model_name == "ResNet50":
            return ResNet50(
                input_size=input_size,
                return_latent_space=args.return_latent_space,
                pretrained=args.pretrained,
                contrastive_learning=args.contrastive_learning,
                latent_dim=latent_dim,
                num_outputs=num_outputs,
                args=args
            )
        elif model_name == "ResNet50-3D":
            return ResNet503D(
                input_size=input_size,
                return_latent_space=args.return_latent_space,
                pretrained=args.pretrained,
                contrastive_learning=args.contrastive_learning,
                latent_dim=latent_dim,
                num_outputs=num_outputs,
                args=args
            )
        elif model_name == "ResNet50-3D-MLP":
            return ResNet503D_MLP(
                input_size=input_size,
                return_latent_space=args.return_latent_space,
                pretrained=args.pretrained,
                contrastive_learning=args.contrastive_learning,
                latent_dim=latent_dim,
                num_outputs=num_outputs,
                args=args
            )
        elif model_name == "ResNet50-4D":
            return ResNet504D(
                input_size=input_size,
                return_latent_space=args.return_latent_space,
                pretrained=args.pretrained,
                contrastive_learning=args.contrastive_learning,
                latent_dim=latent_dim,
                num_outputs=num_outputs,
                args=args
            )
        elif model_name == "SwinTransformer":
            return SwinTransformer(
                input_size=input_size,
                return_latent_space=args.return_latent_space,
                pretrained=args.pretrained,
                contrastive_learning=args.contrastive_learning,
                latent_dim=latent_dim,
                num_outputs=num_outputs,
                args=args
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as e:
        print(f"Error creating model: {e}")
        # Provide fallback with hardcoded values if there's an error
        if model_name == "ResNet50":
            print("Using fallback values for ResNet50")
            return ResNet50(
                input_size=2,
                return_latent_space=False,
                pretrained=False,
                contrastive_learning=False,
                latent_dim=200,
                num_outputs=1,
                args=args
            )
        else:
            raise

