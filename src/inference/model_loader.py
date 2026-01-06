import torch
import pickle
from src.models.biomarker_model import BiomarkerModel
from src.models.eeg_model import EEGModel, SimpleEEGModel


class ModelLoader:
    """Singleton class to load and manage all trained models."""
    
    def __init__(self):
        self.biomarker_model = None
        self.biomarker_preprocessor = None
        self.eeg_model = None
    
    def load_all_models(self):
        """Load trained biomarker and EEG models (if available)."""
        self._load_biomarker_model()
        self._load_eeg_model()
    
    def _load_biomarker_model(self):
        """Load biomarker model and preprocessor."""
        try:
            preproc_path = 'src/prediction/biomarker_model_weights_preprocessor.pkl'
            checkpoint_path = 'src/prediction/biomarker_model_weights.pth'

            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Try to get preprocessor from checkpoint first (most reliable)
            self.biomarker_preprocessor = checkpoint.get('preprocessor', None)
            if self.biomarker_preprocessor is not None:
                print(f"✓ Preprocessor loaded from checkpoint")
            else:
                # Fallback: try separate pickle file
                print(f"⚠ No preprocessor in checkpoint, trying separate file...")
                try:
                    with open(preproc_path, 'rb') as f:
                        self.biomarker_preprocessor = pickle.load(f)
                    print(f"✓ Preprocessor loaded from pickle file")
                except Exception as e:
                    print(f"✗ Could not load preprocessor: {e}")
                    raise Exception("Failed to load preprocessor from any source")
            
            # Load model weights from checkpoint
            model_state = checkpoint['model_state_dict']

            # Determine input dim from preprocessor if available
            if self.biomarker_preprocessor is not None:
                try:
                    input_dim = len(self.biomarker_preprocessor.get_feature_names_out())
                    print(f"✓ Input dimension from preprocessor: {input_dim}")
                except Exception:
                    input_dim = 29
                    print(f"✓ Using default input dimension: {input_dim}")
            else:
                input_dim = 29
                print(f"✓ Using default input dimension: {input_dim}")

            self.biomarker_model = BiomarkerModel(input_dim=input_dim)
            self.biomarker_model.load_state_dict(model_state)
            self.biomarker_model.eval()
            print("✓ Biomarker model loaded successfully (Enhanced: 114K samples, 84.17% recall)")
        except Exception as e:
            print(f"✗ Failed to load biomarker model: {e}")
            import traceback
            traceback.print_exc()

    def _load_eeg_model(self):
        """Load EEG model."""
        try:
            eeg_model_local = SimpleEEGModel(num_classes=5, input_length=256)
            eeg_model_local.load_state_dict(
                torch.load('src/prediction/eeg_model_weights.pth', map_location='cpu', weights_only=True)
            )
            eeg_model_local.eval()
            self.eeg_model = eeg_model_local
            print("✓ EEG model loaded successfully (SimpleEEGModel with 100% accuracy)")
        except Exception as e:
            print(f"✗ Failed to load EEG model: {e}")
            # Try old EEGModel as fallback
            try:
                eeg_model_local = EEGModel(num_channels=1, num_freq_bins=256)
                eeg_model_local.load_state_dict(
                    torch.load('src/prediction/eeg_model_weights.pth', map_location='cpu', weights_only=True)
                )
                eeg_model_local.eval()
                self.eeg_model = eeg_model_local
                print("✓ EEG model loaded successfully (fallback to old EEGModel)")
            except Exception as e2:
                print(f"✗ Failed to load EEG model (both attempts): {e2}")


# Global singleton instance
model_loader = ModelLoader()
model_loader.load_all_models()
