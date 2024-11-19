from .base_engine import BaseEngine
import torch
import numpy as np
from queue import Queue
import os

class StyleTTSEngine(BaseEngine):
    def __init__(self, model_config_path: str, model_checkpoint_path: str, device: str = 'cuda'):
        """
        Initializes the StyleTTS engine.
        Args:
            model_config_path (str): Path to the StyleTTS model config YAML file.
            model_checkpoint_path (str): Path to the pre-trained StyleTTS model checkpoint.
            device (str): Device to run inference on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.queue = Queue()
        self.load_model(model_config_path, model_checkpoint_path)
        self.post_init()

    def post_init(self):
        self.engine_name = "styletts"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for StyleTTS Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 24000 represents 24kHz sample rate.
        """
        import pyaudio
        return pyaudio.paInt16, 1, 24000

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        # Perform inference to generate audio waveform
        audio_waveform = self.inference(text)

        if audio_waveform is not None:
            # Convert waveform to int16 and then to bytes
            audio_data = (audio_waveform * 32767).astype(np.int16).tobytes()
            self.queue.put(audio_data)
            return True
        else:
            return False

    def load_model(self, model_config_path, model_checkpoint_path):
        """
        Loads the StyleTTS model and necessary components.

        Args:
            model_config_path (str): Path to the StyleTTS model config YAML file.
            model_checkpoint_path (str): Path to the pre-trained StyleTTS model checkpoint.
        """
        import yaml
        import torch
        import torchaudio
        from nltk.tokenize import word_tokenize
        from munch import Munch
        from models import build_model
        from utils import load_ASR_models, load_F0_models, recursive_munch
        from text_utils import TextCleaner
        from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
        import numpy as np

        self.textcleaner = TextCleaner()
        self.device = self.device

        # Set random seeds
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        import random
        random.seed(0)

        np.random.seed(0)

        import nltk
        nltk.download('punkt', quiet=True)

        # Initialize mel spectrogram transformation
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        ).to(self.device)
        self.mean, self.std = -4, 4

        # Load model config
        config = yaml.safe_load(open(model_config_path, 'r'))

        # Load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        self.text_aligner = load_ASR_models(ASR_path, ASR_config)

        # Load pretrained F0 model
        F0_path = config.get('F0_path', False)
        self.pitch_extractor = load_F0_models(F0_path)

        # Load BERT model
        from Utils.PLBERT.util import load_plbert
        BERT_path = config.get('PLBERT_dir', False)
        self.plbert = load_plbert(BERT_path)

        # Build the model
        self.model = build_model(recursive_munch(config['model_params']), self.text_aligner, self.pitch_extractor, self.plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        # Load model parameters
        params_whole = torch.load(model_checkpoint_path, map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

        # Initialize the diffusion sampler
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

        # Initialize phonemizer
        import phonemizer
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

        # Set sample rate
        self.sample_rate = 24000

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def inference(self, text: str):
        """
        Run inference on the given text and return the audio waveform.

        Args:
            text (str): Text to synthesize.

        Returns:
            numpy.ndarray: The synthesized audio waveform.
        """
        import torch
        import numpy as np
        from nltk.tokenize import word_tokenize

        # Remove leading/trailing whitespace and quotes from the input text
        text = text.strip()
        text = text.replace('"', '')

        # Use the global phonemizer to convert text to phonemes
        ps = self.global_phonemizer.phonemize([text])

        # Tokenize the phonemized text into words
        ps = word_tokenize(ps[0])

        # Join the tokens back into a single string separated by spaces
        ps = ' '.join(ps)

        # Clean and convert the phonemes to integer tokens using the TextCleaner
        tokens = self.textcleaner(ps)

        # Insert a start token (usually 0) at the beginning of the token list
        tokens.insert(0, 0)

        # Convert the list of tokens to a PyTorch tensor and move it to the appropriate device (CPU or GPU)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Get the length of the token sequence and create a mask for padding (if any)
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = self.length_to_mask(input_lengths).to(tokens.device)

            # Pass the tokens through the text encoder to get text embeddings
            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)

            # Pass the tokens through BERT to get contextualized embeddings for duration prediction
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())

            # Pass the BERT embeddings through another encoder
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            # Generate noise input for diffusion
            noise = torch.randn(1, 1, 256).to(self.device)

            # Use the diffusion sampler to generate style and reference embeddings
            s_pred = self.sampler(
                noise,                              # The noise input for diffusion
                embedding=bert_dur[0].unsqueeze(0), # The BERT embedding as conditioning
                num_steps=5,          # Number of diffusion steps (you can adjust this parameter)
                embedding_scale=1     # Scale for the embedding (you can adjust this parameter)
            ).squeeze(0)

            # Split the predicted embeddings into style (s) and reference (ref) parts
            s = s_pred[:, 128:]   # Style embedding (after the first 128 dimensions)
            ref = s_pred[:, :128] # Reference embedding (first 128 dimensions)

            # Use the predictor's text encoder to combine duration embeddings and style embeddings
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            # Pass the combined embeddings through an LSTM to model temporal dependencies
            x, _ = self.model.predictor.lstm(d)

            # Predict durations for each token
            duration = self.model.predictor.duration_proj(x)

            # Apply sigmoid to normalize durations and sum across the last dimension
            duration = torch.sigmoid(duration).sum(axis=-1)

            # Round the durations to the nearest integer and ensure a minimum duration of 1
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            # Add extra duration to the last token to ensure it's sufficiently long
            pred_dur[-1] += 5

            # Create an alignment target tensor to map tokens to time frames
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().item())).to(self.device)

            # Initialize a frame counter
            c_frame = 0

            # Populate the alignment tensor based on predicted durations
            for i in range(pred_aln_trg.size(0)):
                # Set alignment weights to 1 for the duration of each token
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].item())] = 1
                # Update the frame counter
                c_frame += int(pred_dur[i].item())

            # Encode prosody by combining the duration embeddings and the alignment tensor
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))

            # Predict pitch (F0) and noise components using the prosody encoder
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            # Decode the final audio using the decoder, conditioned on text embeddings, F0, noise, and reference embeddings
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0)), # Aligned text embeddings
                F0_pred,                                       # Predicted pitch
                N_pred,                                        # Predicted noise
                ref.squeeze().unsqueeze(0)                     # Reference embedding
            )

        # Convert the output to waveform
        waveform = out.squeeze().cpu().numpy()

        return waveform
