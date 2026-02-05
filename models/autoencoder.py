"""
Spatio-Temporal Autoencoder for learning normal traffic patterns
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for processing spatio-temporal features
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Combined convolution for input, forget, cell, and output gates
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )
    
    def forward(self, x, hidden_state):
        """
        Args:
            x: Input tensor (batch, channels, height, width)
            hidden_state: Tuple of (h, c) where each is (batch, hidden_channels, height, width)
        """
        h, c = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute gates
        gates = self.conv(combined)
        
        # Split into individual gates
        input_gate, forget_gate, cell_gate, output_gate = torch.split(
            gates, self.hidden_channels, dim=1
        )
        
        # Apply activations
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
        # Update cell state
        c_next = forget_gate * c + input_gate * cell_gate
        
        # Update hidden state
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        """Initialize hidden state"""
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, 
                       device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_channels, height, width,
                       device=self.conv.weight.device)
        )


class SpatioTemporalEncoder(nn.Module):
    """
    Encoder that processes video frames to extract spatio-temporal features
    """
    def __init__(self, input_channels=3, hidden_channels=[64, 128, 256], 
                 kernel_size=3, num_layers=3):
        super(SpatioTemporalEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Spatial feature extraction layers
        self.spatial_layers = nn.ModuleList()
        in_ch = input_channels
        for i in range(num_layers):
            self.spatial_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_channels[i], kernel_size=kernel_size, 
                            stride=2, padding=kernel_size//2),
                    nn.BatchNorm2d(hidden_channels[i]),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = hidden_channels[i]
        
        # Temporal processing with ConvLSTM
        self.temporal_layers = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_layers.append(
                ConvLSTMCell(hidden_channels[i], hidden_channels[i], kernel_size)
            )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, time_steps, channels, height, width)
        Returns:
            Encoded features and hidden states
        """
        batch_size, time_steps, _, height, width = x.size()
        
        # Process each time step
        outputs = []
        hidden_states = None
        
        for t in range(time_steps):
            frame = x[:, t]
            
            # Spatial feature extraction
            spatial_features = []
            h = frame
            for spatial_layer in self.spatial_layers:
                h = spatial_layer(h)
                spatial_features.append(h)
            
            # Temporal processing with ConvLSTM
            if hidden_states is None:
                # Initialize hidden states
                hidden_states = []
                for i, (feat, temporal_layer) in enumerate(zip(spatial_features, self.temporal_layers)):
                    _, _, h_size, w_size = feat.size()
                    hidden_states.append(
                        temporal_layer.init_hidden(batch_size, (h_size, w_size))
                    )
            
            # Update hidden states
            new_hidden_states = []
            for i, (feat, temporal_layer, hidden) in enumerate(
                zip(spatial_features, self.temporal_layers, hidden_states)
            ):
                h_new, c_new = temporal_layer(feat, hidden)
                new_hidden_states.append((h_new, c_new))
            
            hidden_states = new_hidden_states
            outputs.append(hidden_states[-1][0])
        
        # Stack outputs
        encoded = torch.stack(outputs, dim=1)
        
        return encoded, hidden_states


class SpatioTemporalDecoder(nn.Module):
    """
    Decoder that reconstructs video frames from encoded features
    """
    def __init__(self, output_channels=3, hidden_channels=[256, 128, 64],
                 kernel_size=3, num_layers=3):
        super(SpatioTemporalDecoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Temporal processing with ConvLSTM
        self.temporal_layers = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_layers.append(
                ConvLSTMCell(hidden_channels[i], hidden_channels[i], kernel_size)
            )
        
        # Spatial reconstruction layers
        self.spatial_layers = nn.ModuleList()
        for i in range(num_layers):
            out_ch = hidden_channels[i+1] if i < num_layers - 1 else output_channels
            self.spatial_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels[i], out_ch, 
                                     kernel_size=kernel_size, stride=2, 
                                     padding=kernel_size//2, output_padding=1),
                    nn.BatchNorm2d(out_ch) if i < num_layers - 1 else nn.Identity(),
                    nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Sigmoid()
                )
            )
    
    def forward(self, encoded, time_steps):
        """
        Args:
            encoded: Encoded features (batch, time_steps, channels, height, width)
            time_steps: Number of time steps to reconstruct
        Returns:
            Reconstructed frames
        """
        batch_size = encoded.size(0)
        
        # Initialize hidden states
        hidden_states = None
        outputs = []
        
        for t in range(time_steps):
            # Use encoded features if available, otherwise use previous output
            if t < encoded.size(1):
                h = encoded[:, t]
            else:
                h = outputs[-1]
            
            # Initialize hidden states if needed
            if hidden_states is None:
                hidden_states = []
                for i, temporal_layer in enumerate(self.temporal_layers):
                    _, _, h_size, w_size = h.size()
                    # Adjust for upsampling
                    h_size = h_size * (2 ** i)
                    w_size = w_size * (2 ** i)
                    hidden_states.append(
                        temporal_layer.init_hidden(batch_size, (h_size, w_size))
                    )
            
            # Process through temporal and spatial layers
            for i, (temporal_layer, spatial_layer, hidden) in enumerate(
                zip(self.temporal_layers, self.spatial_layers, hidden_states)
            ):
                h, c = temporal_layer(h, hidden)
                hidden_states[i] = (h, c)
                h = spatial_layer(h)
            
            outputs.append(h)
        
        # Stack outputs
        reconstructed = torch.stack(outputs, dim=1)
        
        return reconstructed


class SpatioTemporalAutoencoder(nn.Module):
    """
    Complete autoencoder for learning normal traffic patterns in an unsupervised manner
    """
    def __init__(self, input_channels=3, hidden_channels=[64, 128, 256],
                 kernel_size=3):
        super(SpatioTemporalAutoencoder, self).__init__()
        
        self.encoder = SpatioTemporalEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=len(hidden_channels)
        )
        
        self.decoder = SpatioTemporalDecoder(
            output_channels=input_channels,
            hidden_channels=list(reversed(hidden_channels)),
            kernel_size=kernel_size,
            num_layers=len(hidden_channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input video tensor (batch, time_steps, channels, height, width)
        Returns:
            Reconstructed video
        """
        # Encode
        encoded, hidden_states = self.encoder(x)
        
        # Decode
        time_steps = x.size(1)
        reconstructed = self.decoder(encoded, time_steps)
        
        return reconstructed
    
    def encode(self, x):
        """Extract encoded features"""
        encoded, _ = self.encoder(x)
        return encoded
