import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, Any
import tensorflow_hub as hub
import tensorflow as tf
from transformers import Wav2Vec2Model, Wav2Vec2Config

# Register the problematic global to allow safe deserialization.
# import torch.serialization
# torch.serialization.add_safe_globals(["__path__._path"])

class YAMNetDecoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.target_length = int(cfg.yamnet_duration * cfg.sample_rate)  # 10 sec * 16000 Hz = 160000 samples
        
    def get_audio(self, file_path: str) -> np.ndarray:
        """Load and normalize audio with fixed 10-second duration"""
        try:
            # Load audio
            audio, sr = librosa.load(
                file_path, 
                sr=self.cfg.sample_rate, 
                mono=True
            )
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Handle duration
            if len(audio) < self.target_length:
                # Pad shorter audio with zeros
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                # For longer audio, take the middle segment
                start = (len(audio) - self.target_length) // 2
                audio = audio[start:start + self.target_length]
            
            return audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    def frame_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convert fixed-length audio into frames for YAMNet"""
        # YAMNet uses 25ms frames (15360 samples at 16kHz)
        frame_length = 15360
        
        # Calculate number of complete frames
        n_frames = self.target_length // frame_length
        
        target_length = n_frames * frame_length
        
        # Pad or trim audio to match target length
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Reshape into frames
        frames = audio.reshape(n_frames, frame_length)
        return frames

class Wav2Vec2Decoder:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def get_audio(self, file_path: str) -> np.ndarray:
        """Load and normalize audio"""
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=self.cfg.sample_rate, 
                mono=True
            )
            # Normalize
            audio = librosa.util.normalize(audio)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return np.zeros(self.cfg.sample_rate)
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process audio for Wav2Vec2"""
        target_length = int(self.cfg.wav2vec_duration * self.cfg.sample_rate)
        
        if len(audio) > target_length:
            # Take center segment
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        else:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
        return audio  # Shape: (target_length,)
    
class YAMNetFeatureExtractor:
    """Efficient YAMNet feature extraction for batched PyTorch tensors."""
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

    def extract_features(self, waveform_batch):
        """Extracts YAMNet embeddings efficiently for a batch of PyTorch tensors."""
        with torch.no_grad():
            batch_size = waveform_batch.shape[0]
            embeddings_list = []
            
            for i in range(batch_size):
                # Get single waveform and flatten it to 1D
                single_waveform = waveform_batch[i].cpu().flatten().numpy()
                
                # Extract features using YAMNet (expects 1D input)
                _, embeddings, _ = self.model(single_waveform)
                embeddings_list.append(embeddings.numpy())
            
            # Stack embeddings and convert to PyTorch tensor
            embeddings_torch = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
            return embeddings_torch.to(waveform_batch.device)

class YAMNetFineTuner(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.yamnet_extractor = YAMNetFeatureExtractor()
        
        # Define trainable classifier layers with proper BatchNorm momentum
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, momentum=0.01),  # Lower momentum for stable training
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights with proper scaling"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input validation
        if x.dim() != 3:  # (batch, frames, samples)
            raise ValueError(f"Expected 3D input (batch, frames, samples), got shape {x.shape}")
            
        # Extract features using frozen YAMNet
        features = self.yamnet_extractor.extract_features(x)  # (batch, frames, 1024)
        
        # Global average pooling with safe handling
        if features.dim() == 3:
            pooled = torch.mean(features, dim=1)  # (batch, 1024)
        else:
            raise ValueError(f"Expected 3D features, got shape {features.shape}")
        
        # Classification
        output = self.classifier(pooled)
        return output

class Wav2Vec2FineTuner(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_encoder=True):
        super().__init__()
        
        # Load pretrained Wav2Vec2 model
        if pretrained:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        else:
            config = Wav2Vec2Config()
            self.wav2vec2 = Wav2Vec2Model(config)
        
        # Initial freezing
        if freeze_encoder:
            self.freeze_feature_extractor()
            self.freeze_encoder_layers(12)  # Freeze all layers initially
        
        hidden_size = self.wav2vec2.config.hidden_size
        
        # Improved classifier architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights properly"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # Use scaled initialization for GELU
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def freeze_feature_extractor(self):
        """Freeze the CNN feature extractor"""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
    
    def freeze_encoder_layers(self, num_layers):
        """Freeze specified number of transformer layers"""
        for layer in self.wav2vec2.encoder.layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder_layers(self, num_layers_to_unfreeze):
        """Gradually unfreeze layers from top"""
        total_layers = len(self.wav2vec2.encoder.layers)
        for i, layer in enumerate(reversed(self.wav2vec2.encoder.layers)):
            if i < num_layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        # Input validation
        if x.dim() != 2:  # (batch_size, sequence_length)
            raise ValueError(f"Expected input shape (batch_size, sequence_length), got {x.shape}")

        # Wav2Vec2 forward pass
        outputs = self.wav2vec2(x)
        hidden_states = outputs.last_hidden_state
        
        # Global mean pooling
        pooled = torch.mean(hidden_states, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output

class AudioInference:
    def __init__(self, cfg):
        """Initialize inference with config and model paths"""
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.yamnet_models = []
        self.wav2vec2_models = []
        
        # Load YAMNet models
        for i in range(5):
            model = YAMNetFineTuner(num_classes=cfg.num_classes, cfg=cfg).to(self.device)
            checkpoint = torch.load(f'models/best_yamnet_fold_{i}_model.pt', map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.yamnet_models.append(model)
        
        # Load Wav2Vec2 models
        for i in range(5):
            model = Wav2Vec2FineTuner(num_classes=cfg.num_classes).to(self.device)
            checkpoint = torch.load(f'models/best_wav2vec2_fold_{i}_model.pt', map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.wav2vec2_models.append(model)

    def preprocess_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Preprocess audio for both models"""
        # Load and normalize audio
        audio, _ = librosa.load(audio_path, sr=self.cfg.sample_rate, mono=True)
        audio = librosa.util.normalize(audio)
        
        # YAMNet preprocessing
        yamnet_decoder = YAMNetDecoder(self.cfg)
        yamnet_audio = yamnet_decoder.get_audio(audio_path)
        yamnet_frames = yamnet_decoder.frame_audio(yamnet_audio)
        yamnet_input = torch.from_numpy(yamnet_frames).float()
        
        # Wav2Vec2 preprocessing
        wav2vec2_decoder = Wav2Vec2Decoder(self.cfg)
        wav2vec2_audio = wav2vec2_decoder.process_audio(audio)
        wav2vec2_input = torch.from_numpy(wav2vec2_audio).float()
        
        return {
            'yamnet': yamnet_input.unsqueeze(0),  # Add batch dimension
            'wav2vec2': wav2vec2_input.unsqueeze(0)  # Add batch dimension
        }

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Predict class for single audio file"""
        # Preprocess audio
        inputs = self.preprocess_audio(audio_path)
        
        # YAMNet predictions
        yamnet_probs = []
        for model in self.yamnet_models:
            logits = model(inputs['yamnet'].to(self.device))
            probs = F.softmax(logits, dim=1)
            yamnet_probs.append(probs)
        yamnet_mean_probs = torch.stack(yamnet_probs).mean(dim=0)
        
        # Wav2Vec2 predictions
        wav2vec2_probs = []
        for model in self.wav2vec2_models:
            logits = model(inputs['wav2vec2'].to(self.device))
            probs = F.softmax(logits, dim=1)
            wav2vec2_probs.append(probs)
        wav2vec2_mean_probs = torch.stack(wav2vec2_probs).mean(dim=0)
        
        # Ensemble predictions
        ensemble_probs = (yamnet_mean_probs + wav2vec2_mean_probs) / 2
        pred_class = torch.argmax(ensemble_probs, dim=1).item()
        
        # Get probabilities for each class
        class_probs = {
            'yamnet': yamnet_mean_probs.cpu().numpy()[0],
            'wav2vec2': wav2vec2_mean_probs.cpu().numpy()[0],
            'ensemble': ensemble_probs.cpu().numpy()[0]
        }
        
        # Map prediction to class name
        class_name = self.cfg.int2label[pred_class]
        
        return {
            'predicted_class': class_name,
            'class_id': pred_class,
            'probabilities': class_probs
        }

def main():
    """Example usage"""
    # Initialize config
    class CFG:
        sample_rate = 16000
        yamnet_frame_duration = 0.96
        yamnet_hop_duration = 0.48
        wav2vec_duration = 10.0
        yamnet_duration = 10.0
        num_classes = 3
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        int2label = {
            0: 'infant_cry',
            1: 'scream',
            2: 'normal_utterance'
        }
        
        # Add other necessary parameters from your training config
        yamnet_samples_per_frame = int(yamnet_frame_duration * sample_rate)
        yamnet_hop_samples = int(yamnet_hop_duration * sample_rate)
        n_mels = 64
        window_size = int(0.025 * sample_rate)
        hop_length = int(0.01 * sample_rate)
        fmin = 125
        fmax = 7500

    # Initialize inference
    cfg = CFG()
    inferencer = AudioInference(cfg)
    
    # Example prediction
    audio_path = "path/to/your/audio.wav"
    result = inferencer.predict(audio_path)
    
    # Print results
    print(f"\nPrediction Results for {audio_path}")
    print(f"Predicted Class: {result['predicted_class']}")
    print("\nClass Probabilities:")
    for model_name, probs in result['probabilities'].items():
        print(f"\n{model_name.capitalize()} Model:")
        for i, prob in enumerate(probs):
            class_name = cfg.int2label[i]
            print(f"{class_name}: {prob:.4f}")

if __name__ == "__main__":
    main()
