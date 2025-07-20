import sys
import os
import torch
import torchaudio
import numpy as np
import json
import logging
from typing import Optional, Union

# RealtimeTTS imports
from .base_engine import BaseEngine

class ZipVoiceVoice:
    """
    Represents a ZipVoice voice, defined by a prompt audio file and its transcription.
    """
    def __init__(self, prompt_wav_path: str, prompt_text: str):
        """
        Initializes a ZipVoice voice configuration.

        Args:
            prompt_wav_path (str): Path to the reference audio file to be mimicked.
            prompt_text (str): The exact transcription of the reference audio file.
        """
        if not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"Prompt WAV file not found at: {prompt_wav_path}")
        self.prompt_wav_path = prompt_wav_path
        self.prompt_text = prompt_text

    def __repr__(self):
        return f"ZipVoiceVoice(prompt_wav='{self.prompt_wav_path}', prompt_text='{self.prompt_text}')"


class ZipVoiceEngine(BaseEngine):
    """
    ZipVoice Text-to-Speech Engine for RealtimeTTS.

    This engine uses the ZipVoice model to synthesize speech based on a voice prompt.
    """
    def __init__(self,
                 zipvoice_root: str,
                 voice: ZipVoiceVoice,
                 model_name: str = "zipvoice",
                 checkpoint: Optional[str] = None,
                 model_config: Optional[str] = None,
                 vocoder_path: Optional[str] = None,
                 token_file: Optional[str] = None,
                 tokenizer_type: str = "emilia",
                 device: str = 'cuda',
                 # Inference parameters
                 speed: float = 1.0,
                 guidance_scale: Optional[float] = None,
                 num_step: Optional[int] = None,
                 t_shift: float = 0.5,
                 target_rms: float = 0.1,
                 feat_scale: float = 0.1
                 ):
        """
        Initializes the ZipVoice engine.

        Args:
            zipvoice_root (str):
                Path to the root directory of the ZipVoice repository.
            voice (ZipVoiceVoice):
                A ZipVoiceVoice object containing the prompt audio and text.
            model_name (str, optional):
                The model to use, either "zipvoice" or "zipvoice_distill".
                Defaults to "zipvoice".
            checkpoint (Optional[str], optional):
                Path to a local model checkpoint file (.pt or .safetensors).
                If None, downloads from Hugging Face. Defaults to None.
            model_config (Optional[str], optional):
                Path to a local model config JSON file.
                If None, downloads from Hugging Face. Defaults to None.
            vocoder_path (Optional[str], optional):
                Path to a local vocoder directory.
                If None, downloads from Hugging Face. Defaults to None.
            token_file (Optional[str], optional):
                Path to a local tokens.txt file.
                If None, downloads from Hugging Face. Defaults to None.
            tokenizer_type (str, optional):
                Type of tokenizer to use ('emilia', 'libritts', 'espeak', 'simple').
                Defaults to "emilia".
            device (str, optional):
                Device to run inference on ('cuda', 'mps', 'cpu').
                Defaults to 'cuda'.
            speed (float, optional):
                Speech speed control. >1.0 is faster. Defaults to 1.0.
            guidance_scale (Optional[float], optional):
                Classifier-free guidance scale. If None, uses model default.
                Defaults to None.
            num_step (Optional[int], optional):
                Number of sampling steps. If None, uses model default.
                Defaults to None.
            t_shift (float, optional):
                Time shift parameter. Defaults to 0.5.
            target_rms (float, optional):
                Target RMS for prompt normalization. Set to 0 to disable.
                Defaults to 0.1.
            feat_scale (float, optional):
                Scale factor for fbank features. Defaults to 0.1.
        """
        # 1. Add zipvoice_root to sys.path to allow imports
        self.zipvoice_root = zipvoice_root.replace("\\", "/")
        if not os.path.isdir(self.zipvoice_root):
            raise NotADirectoryError(f"zipvoice_root is not a valid directory: {self.zipvoice_root}")
        sys.path.insert(0, os.path.abspath(self.zipvoice_root))

        # 2. Import dependencies now that the path is set
        try:
            from huggingface_hub import hf_hub_download
            from vocos import Vocos
            from zipvoice.models.zipvoice import ZipVoice
            from zipvoice.models.zipvoice_distill import ZipVoiceDistill
            from zipvoice.tokenizer.tokenizer import (
                EmiliaTokenizer, EspeakTokenizer, LibriTTSTokenizer, SimpleTokenizer
            )
            from zipvoice.utils.checkpoint import load_checkpoint
            from zipvoice.utils.feature import VocosFbank
            import safetensors.torch
        except ImportError as e:
            raise ImportError(f"Failed to import ZipVoice dependencies. Ensure '{self.zipvoice_root}' is the correct path to the ZipVoice project. Error: {e}")

        # 3. Initialize parameters
        self.voice = voice
        self.model_name = model_name
        self.speed = speed
        self.t_shift = t_shift
        self.target_rms = target_rms
        self.feat_scale = feat_scale

        # Set device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"ZipVoiceEngine: Using device: {self.device}")

        # 4. Set model-specific defaults for num_step and guidance_scale
        model_defaults = {
            "zipvoice": {"num_step": 16, "guidance_scale": 1.0},
            "zipvoice_distill": {"num_step": 8, "guidance_scale": 3.0},
        }
        model_specific_defaults = model_defaults.get(self.model_name, {})
        self.num_step = num_step if num_step is not None else model_specific_defaults.get('num_step')
        self.guidance_scale = guidance_scale if guidance_scale is not None else model_specific_defaults.get('guidance_scale')

        # 5. Load components (model, tokenizer, vocoder)
        HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
        PRETRAINED_MODEL_PATHS = {"zipvoice": "zipvoice/model.pt", "zipvoice_distill": "zipvoice_distill/model.pt"}
        TOKEN_FILE_PATHS = {"zipvoice": "zipvoice/tokens.txt", "zipvoice_distill": "zipvoice_distill/tokens.txt"}
        MODEL_CONFIG_PATHS = {"zipvoice": "zipvoice/zipvoice_base.json", "zipvoice_distill": "zipvoice_distill/zipvoice_base.json"}

        logging.info("Loading ZipVoice model components...")
        model_config_path = hf_hub_download(HUGGINGFACE_REPO, filename=MODEL_CONFIG_PATHS[self.model_name]) if model_config is None else model_config
        with open(model_config_path, "r") as f:
            self.model_config_data = json.load(f)
        self.sampling_rate = self.model_config_data["feature"]["sampling_rate"]

        token_file_path = hf_hub_download(HUGGINGFACE_REPO, filename=TOKEN_FILE_PATHS[self.model_name]) if token_file is None else token_file
        tokenizer_map = {
            "emilia": EmiliaTokenizer, "libritts": LibriTTSTokenizer,
            "espeak": EspeakTokenizer, "simple": SimpleTokenizer
        }
        self.tokenizer = tokenizer_map[tokenizer_type](token_file=token_file_path)
        tokenizer_config = {"vocab_size": self.tokenizer.vocab_size, "pad_id": self.tokenizer.pad_id}

        model_ckpt_path = hf_hub_download(HUGGINGFACE_REPO, filename=PRETRAINED_MODEL_PATHS[self.model_name]) if checkpoint is None else checkpoint
        model_class = ZipVoice if self.model_name == "zipvoice" else ZipVoiceDistill
        self.model = model_class(**self.model_config_data["model"], **tokenizer_config)

        if model_ckpt_path.endswith(".safetensors"):
            safetensors.torch.load_model(self.model, model_ckpt_path)
        else:
            load_checkpoint(filename=model_ckpt_path, model=self.model, strict=True)
        self.model.to(self.device).eval()

        if vocoder_path:
            self.vocoder = Vocos.from_hparams(f"{vocoder_path}/config.yaml")
            state_dict = torch.load(f"{vocoder_path}/pytorch_model.bin", weights_only=True, map_location="cpu")
            self.vocoder.load_state_dict(state_dict)
        else:
            self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocoder.to(self.device).eval()

        self.feature_extractor = VocosFbank()
        logging.info("ZipVoiceEngine initialized successfully.")
        
        self.post_init()

    def post_init(self):
        self.engine_name = "zipvoice"

    def get_stream_info(self):
        import pyaudio
        # The vocoder outputs float32, but we convert to int16 for broader compatibility.
        return pyaudio.paInt16, 1, self.sampling_rate

    @torch.inference_mode()
    def synthesize(self, text: str) -> bool:
        super().synthesize(text)
        try:
            prompt_text = self.voice.prompt_text
            prompt_wav_path = self.voice.prompt_wav_path

            tokens = self.tokenizer.texts_to_token_ids([text])
            prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])

            prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav_path)
            if prompt_sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=prompt_sampling_rate, new_freq=self.sampling_rate)
                prompt_wav = resampler(prompt_wav)

            prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
            if self.target_rms > 0 and prompt_rms < self.target_rms:
                prompt_wav = prompt_wav * self.target_rms / prompt_rms

            prompt_features = self.feature_extractor.extract(prompt_wav, sampling_rate=self.sampling_rate).to(self.device)
            prompt_features = prompt_features.unsqueeze(0) * self.feat_scale
            prompt_features_lens = torch.tensor([prompt_features.size(1)], device=self.device)

            (pred_features, _, _, _) = self.model.sample(
                tokens=tokens, prompt_tokens=prompt_tokens,
                prompt_features=prompt_features, prompt_features_lens=prompt_features_lens,
                speed=self.speed, t_shift=self.t_shift, duration="predict",
                num_step=self.num_step, guidance_scale=self.guidance_scale,
            )

            pred_features = pred_features.permute(0, 2, 1) / self.feat_scale
            wav_float = self.vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

            if self.target_rms > 0 and prompt_rms < self.target_rms:
                wav_float = wav_float * prompt_rms / self.target_rms

            audio_data = (wav_float.cpu().numpy() * 32767).astype(np.int16).tobytes()
            self.queue.put(audio_data)
            return True
        except Exception as e:
            logging.error(f"ZipVoice synthesis error: {e}")
            return False

    def set_voice(self, voice: ZipVoiceVoice):
        if isinstance(voice, ZipVoiceVoice):
            self.voice = voice
            logging.info(f"ZipVoiceEngine: Voice updated to {voice}")
        else:
            raise TypeError("Voice must be an instance of ZipVoiceVoice.")

    def get_voices(self):
        # ZipVoice uses prompts, not a list of pre-defined voices.
        # The user is expected to create ZipVoiceVoice objects manually.
        return []

    def set_voice_parameters(self, **voice_parameters):
        valid_params = ['speed', 'guidance_scale', 'num_step', 't_shift']
        for param, value in voice_parameters.items():
            if param in valid_params:
                setattr(self, param, value)
                logging.info(f"ZipVoiceEngine: Set {param} to {value}")

    def shutdown(self):
        # Clean up GPU memory
        del self.model
        del self.vocoder
        del self.tokenizer
        del self.feature_extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Remove the path we added
        sys.path.remove(os.path.abspath(self.zipvoice_root))
        logging.info("ZipVoiceEngine shutdown.")