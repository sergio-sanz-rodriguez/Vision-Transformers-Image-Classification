import torch
import torchvision
from torch import nn
from torch.nn.init import trunc_normal_

# Implementation of a vision transformer following the paper "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        emb_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 1. Initialize the class with appropriate variables
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 emb_dim:int=768,
                 emb_dropout:float=0.1):
        super().__init__()

        # 2. Make the image size is divisble by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 3. Create a layer to turn an image into patches
        self.conv_proj = nn.Conv2d(in_channels=in_channels,
                                   out_channels=emb_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector. dim=0 is the batch size, dim=1 is the embedding dimension.
                                  end_dim=3)
        
        # 5. Token embedding class
        self.class_token = trunc_normal_(nn.Parameter(torch.zeros(1, 1, emb_dim), requires_grad=True), std=0.02)
                       
        # 6. Position embedding class
        num_patches = (img_size * img_size) // patch_size**2
        self.pos_embedding = trunc_normal_(nn.Parameter(torch.zeros(1, num_patches+1, emb_dim), requires_grad=True), std=0.02)
        
        # 7. Create embedding dropout value
        self.emb_dropout = nn.Dropout(p=emb_dropout)

    # 8. Define the forward method
    def forward(self, x):
        
        # 9. Linear projection of patches 
        x = self.conv_proj(x)

        # 10. Flatten the linear transformed patches
        x = self.flatten(x)

        # 11. Adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        x = x.permute(0, 2, 1)

        # 12. Create class token and prepend
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        
        # 13. Create position embedding
        x = x + self.pos_embedding

        # 14. Run embedding dropout (Appendix B.1)              
        x = self.emb_dropout(x)

        return x


class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 1. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 2. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

        # 3. Create the Multi-Head Attention (MSA) layer
        self.self_attention = nn.MultiheadAttention(embed_dim=emb_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True) # does our batch dimension come first?

    # 4. Create a forward() method to pass the data throguh the layers
    def forward(self, x):

        # 5. Normalization layer
        x = self.layer_norm(x)

        # 6. Multihead attention layer
        x, _ = self.self_attention(query=x, # query embeddings
                                   key=x, # key embeddings
                                   value=x, # value embeddings
                                   need_weights=False) # do we need the weights or just the layer outputs?
        return x
    

class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 emb_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base (X4)
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 2. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

        # 3. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=emb_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity"
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=emb_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer"
        )

    # 4. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        return self.mlp(self.layer_norm(x))
    

class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 attn_dropout:float=0, # Amount of dropout for attention layers
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base                 
                 ): 
        super().__init__()

        # 2. Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(emb_dim=emb_dim,
                                                     num_heads=num_heads,
                                                     dropout=attn_dropout)

        # 3. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(emb_dim=emb_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    # 4. Create a forward() method
    def forward(self, x):

        # 5. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x

        # 6. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x
    

# Create a ViT class that inherits from nn.Module
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 emb_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        
        """
        Initializes a Vision Transformer (ViT) model with specified hyperparameters.

        The constructor sets up the ViT model by configuring the input image size, number of transformer layers,
        embedding dimension, number of attention heads, MLP size, and dropout rates, based on the ViT-Base configuration 
        as detailed in the original ViT paper. These parameters are also customizable to suit different downstream tasks.

        Args:
        - img_size (int, optional): The resolution of the input images. Default is 224.
        - in_channels (int, optional): The number of input image channels. Default is 3 (RGB).
        - patch_size (int, optional): The size of patches to divide the input image into. Default is 16.
        - num_transformer_layers (int, optional): The number of transformer layers. Default is 12 for ViT-Base.
        - emb_dim (int, optional): The dimensionality of the embedding space. Default is 768 for ViT-Base.
        - mlp_size (int, optional): The size of the MLP hidden layers. Default is 3072 for ViT-Base.
        - num_heads (int, optional): The number of attention heads in each transformer layer. Default is 12.
        - attn_dropout (float, optional): The dropout rate applied to attention layers. Default is 0.
        - mlp_dropout (float, optional): The dropout rate applied to the MLP layers. Default is 0.1.
        - emb_dropout (float, optional): The dropout rate applied to patch and position embeddings. Default is 0.1.
        - num_classes (int, optional): The number of output classes. Default is 1000 for ImageNet, but can be customized.

        Note:
        This initialization is based on the ViT-Base/16 model as described in the Vision Transformer paper. Custom values can
        be provided for these parameters based on the specific task or dataset.
        """

        super().__init__() # don't forget the super().__init__()!

        # 2. Create patch embedding layer
        self.embedder = PatchEmbedding(img_size=img_size,
                                        in_channels=in_channels,
                                        patch_size=patch_size,
                                        emb_dim=emb_dim,
                                        emb_dropout=emb_dropout)
        
        #self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)

        #num_patches = (img_size * img_size) // patch_size**2 # N = HW/P^2
        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim), requires_grad=True)

        # 3. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(emb_dim=emb_dim,
                                                               num_heads=num_heads,
                                                               mlp_size=mlp_size,
                                                               attn_dropout=attn_dropout,
                                                               mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        
        # 4. Create classifier head
        layer_norm = nn.LayerNorm(normalized_shape=emb_dim)
        head = nn.Linear(in_features=emb_dim, out_features=num_classes)
        self.classifier = nn.Sequential(
            layer_norm,
            head
        )

    def copy_weights(self,
                      model_weights: torchvision.models.ViT_B_16_Weights):
        """
        Copies the pretrained weights from a ViT model (Vision Transformer) to the current model.
        This method assumes that the current model has a structure compatible with the ViT-base architecture.
        
        Args:
            model_weights (torchvision.models.ViT_B_16_Weights): The pretrained weights of the ViT model.
                This should be a state dictionary from a ViT-B_16 architecture, such as the one provided
                by torchvision's ViT_B_16_Weights.DEFAULT.

        Notes:
            - This method manually copies weights from the pretrained ViT model to the corresponding layers of the current model.
            - It supports the ViT-base architecture with 12 transformer encoder layers and expects a similar
            structure in the target model (e.g., embedder, encoder layers, classifier).
            - This method does not update the optimizer state or any other model parameters beyond the weights.

        Example:
            pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            model.copy_weights(pretrained_vit_weights)
        """


        # Get the current state_dict of ViT
        state_dict = self.state_dict()

        # Get the actual model weights from the input model
        pretrained_state_dict = model_weights.get_state_dict()

        # Update the parameters element-wise
        state_dict['embedder.class_token'].copy_(pretrained_state_dict['class_token'])
        state_dict['embedder.pos_embedding'].copy_(pretrained_state_dict['encoder.pos_embedding'])
        state_dict['embedder.conv_proj.weight'].copy_(pretrained_state_dict['conv_proj.weight'])
        state_dict['embedder.conv_proj.bias'].copy_(pretrained_state_dict['conv_proj.bias'])

        # Dynamically get the number of encoder layers from model_weights
        encoder_layer_keys = [key for key in pretrained_state_dict.keys() if 'encoder.layers' in key]
        num_encoder_layers = len(set([key.split('.')[2] for key in encoder_layer_keys]))

        # Update encoder layers
        for layer in range(num_encoder_layers):
            state_dict[f'encoder.{layer}.msa_block.layer_norm.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_1.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.layer_norm.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_1.bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_2.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_2.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_1.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_1.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_2.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_2.bias']
            )

        # Reload updated state_dict into the model
        self.load_state_dict(state_dict)

 
    def copy_weights_bk(self,
                     model_weights:torch.nn.Module
                     ):
        """
        Copies pretrained weights from a compatible Vision Transformer (ViT) model into the current model.

        This function facilitates transfer learning by reusing pretrained weights to initialize the current model for
        fine-tuning or other downstream tasks.

        Compatibility:
        - Assumes strict compatibility between the layer structure and parameter naming conventions of the source
            and target models.

        Arguments:
        - model_weights (torch.nn.Module): A pretrained ViT-Base/16 model whose weights will be copied.

        Behavior:
        - Transfers weights for the embedder, encoder layers, and classifier head.
        - Updates parameters element-wise to ensure consistency between source and target models.

        Note:
        Differences in layer structure or naming conventions may result in errors during the copying process.
        """

        # Get the current state_dict of vitbase
        state_dict = self.state_dict()  

        # Update the parameters element-wise
        state_dict['embedder.class_token'].copy_(model_weights.state_dict()['class_token'])
        state_dict['embedder.pos_embedding'].copy_(model_weights.state_dict()['encoder.pos_embedding'])
        state_dict['embedder.conv_proj.weight'].copy_(model_weights.state_dict()['conv_proj.weight'])
        state_dict['embedder.conv_proj.bias'].copy_(model_weights.state_dict()['conv_proj.bias'])

        # Dynamically get the number of encoder layers from model_weights
        num_encoder_layers = len(model_weights.encoder.layer)

        # Update encoder layers
        for layer in range(num_encoder_layers):
            state_dict[f'encoder.{layer}.msa_block.layer_norm.weight'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.ln_1.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.layer_norm.bias'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.ln_1.bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_weight'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_bias'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.weight'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.bias'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.weight'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.ln_2.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.bias'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.ln_2.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.weight'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.mlp.0.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.bias'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.mlp.0.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.weight'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.mlp.3.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.bias'].copy_(
                model_weights.state_dict()[f'encoder.layers.encoder_layer_{layer}.mlp.3.bias']
            )

        # Update classifier
        state_dict['classifier.0.weight'].copy_(model_weights.state_dict()['encoder.ln.weight'])
        state_dict['classifier.0.bias'].copy_(model_weights.state_dict()['encoder.ln.bias'])
        state_dict['classifier.1.weight'].copy_(model_weights.state_dict()['heads.weight'])
        state_dict['classifier.1.bias'].copy_(model_weights.state_dict()['heads.bias'])

        # Reload updated state_dict into the model
        self.load_state_dict(state_dict)

    def set_params_frozen(self,                          
                          except_head:bool=True):
        """
        Freezes parameters of different components, allowing exceptions.

        Args:        
            except_head (bool): If True, excludes the classifier head from being frozen.
        """

        for param in self.embedder.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = except_head

    # 5. Create a forward() method
    def forward(self, x):

        # 6. Create patch embedding (equation 1)
        x = self.embedder(x)

        # 7. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.encoder(x)

        # 8. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x
    

# Create a ViT class that inherits from nn.Module using already implemented transformer classes from torchvision
class ViTv2(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 emb_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        # 3. Create patch embedding layer
        self.embedder = PatchEmbedding(img_size=img_size,
                                        in_channels=in_channels,
                                        patch_size=patch_size,
                                        emb_dim=emb_dim,
                                        emb_droput=emb_dropout)
        
        # 4. Create a single transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
                                                        nhead=num_heads,
                                                        dim_feedforward=mlp_size,
                                                        dropout=mlp_dropout,
                                                        activation="gelu",
                                                        batch_first=True,
                                                        norm_first=True)
        
        # 5. Crete the stacked transformer encoder
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=num_transformer_layers)
        
        # 6. Create classifier head
        layer_norm = nn.LayerNorm(normalized_shape=emb_dim)
        head = nn.Linear(in_features=emb_dim, out_features=num_classes)
        self.classifier = nn.Sequential(
            layer_norm,
            head
        )

    # 7. Create a forward() method
    def forward(self, x):

        # 8. Create patch embedding (equation 1)
        #x = x + self.position_embedding[:, :x.size(1), :]
        x = self.embedder(x)

        # 10. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.encoder(x)

        # 11. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x