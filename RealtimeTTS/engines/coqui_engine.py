from multiprocessing import Process, Pipe, Event
from .base_engine import BaseEngine
from threading import Lock
from tqdm import tqdm
import numpy as np
import traceback
import requests
import logging
import pyaudio
import torch
import json
import os
import re

class CoquiEngine(BaseEngine):

    def __init__(self, 
                 model_name = "tts_models/multilingual/multi-dataset/xtts_v2",
                 specific_model = "2.0.2",
                 local_models_path = None, # specify a global model path here (otherwise it will create a directory "models" in the script directory)
                 voices_path = None,
                 cloning_reference_wav: str = "",
                 language = "en",
                 speed = 1.0,
                 thread_count = 6,
                 stream_chunk_size = 20,
                 overlap_wav_len = 1024,
                 temperature = 0.85,
                 length_penalty = 1.0,
                 repetition_penalty = 7.0,
                 top_k = 50,
                 top_p = 0.85,
                 enable_text_splitting = True,
                 full_sentences = False,
                 level=logging.WARNING,
                 use_mps = False,
                 use_deepspeed = False,
                 prepare_text_for_synthesis_callback = None,
                 ):
        """
        Initializes a coqui voice realtime text to speech engine object.

        Args:
            model_name (str): Name of the coqui model to use. Tested with XTTS only.
            specific_model (str): Name of the specific model to use. If not specified, the most recent model will be downloaded.
            local_models_path (str): Path to a local models directory. If not specified, a directory "models" will be created in the script directory.
            cloning_reference_wav (str): Name to the file containing the voice to clone. Works with a 44100Hz or 22050Hz mono 32bit float WAV file.
            language (str): Language to use for the coqui model.
            speed (float): Speed factor for the coqui model.
            thread_count (int): Number of threads to use for the coqui model.
            stream_chunk_size (int): Chunk size for the coqui model.
            overlap_wav_len (int): Overlap length for the coqui model.
            temperature (float): Temperature for the coqui model.
            length_penalty (float): Length penalty for the coqui model.
            repetition_penalty (float): Repetition penalty for the coqui model.
            top_k (int): Top K for the coqui model. 
            top_p (float): Top P for the coqui model.
            enable_text_splitting (bool): Enable text splitting for the coqui model.
            full_sentences (bool): Enable full sentences for the coqui model.
            level (int): Logging level for the coqui model.
            use_mps (bool): Enable MPS for the coqui model.
            use_deepspeed (bool): Enable deepspeed for the coqui model.
            prepare_text_for_synthesis_callback (function): Function to prepare text for synthesis. If not specified, a default sentence parser will be used. 
        """

        self._synthesize_lock = Lock()
        self.model_name = model_name
        self.language = language
        self.cloning_reference_wav = cloning_reference_wav
        self.speed = speed
        self.specific_model = specific_model
        if not local_models_path:
            local_models_path = os.environ.get("COQUI_MODEL_PATH")
            if local_models_path and len(local_models_path) > 0:
                logging.info(f"Local models path from environment variable COQUI_MODEL_PATH: \"{local_models_path}\"")
        self.local_models_path = local_models_path
        self.prepare_text_for_synthesis_callback = prepare_text_for_synthesis_callback

        # Start the worker process
        self.main_synthesize_ready_event = Event()
        self.parent_synthesize_pipe, child_synthesize_pipe = Pipe()
        self.voices_path = voices_path

        # download coqui model
        self.local_model_path = None
        if not self.specific_model:
            from TTS.utils.manage import ModelManager
            logging.info(f"Download most recent XTTS Model if available")
            ModelManager().download_model(model_name)
        else:
            logging.info(f"Local XTTS Model: \"{specific_model}\" specified")
            self.local_model_path = self.download_model(specific_model, local_models_path)

        self.synthesize_process = Process(target=CoquiEngine._synthesize_worker, args=(child_synthesize_pipe, model_name, cloning_reference_wav, language, self.main_synthesize_ready_event, level, self.speed, thread_count, stream_chunk_size, full_sentences, overlap_wav_len, temperature, length_penalty, repetition_penalty, top_k, top_p, enable_text_splitting, use_mps, self.local_model_path, use_deepspeed, self.voices_path))
        self.synthesize_process.start()

        logging.debug('Waiting for coqui text to speech synthesize model to start')
        self.main_synthesize_ready_event.wait()
        logging.info('Coqui synthesis model ready')

    def post_init(self):
        self.engine_name = "coqui"

    @staticmethod
    def _synthesize_worker(conn, model_name, cloning_reference_wav, language, ready_event, loglevel, speed, thread_count, stream_chunk_size, full_sentences, overlap_wav_len, temperature, length_penalty, repetition_penalty, top_k, top_p, enable_text_splitting, use_mps, local_model_path, use_deepspeed, voices_path):
        """
        Worker process for the coqui text to speech synthesis model.

        Args:
            conn (multiprocessing.Connection): Connection to the parent process.
            model_name (str): Name of the coqui model to use.
            cloning_reference_wav (str): Name to the file containing the voice to clone.
            language (str): Language to use for the coqui model.
            ready_event (multiprocessing.Event): Event to signal when the model is ready.
        """

        from TTS.utils.generic_utils import get_user_data_dir
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.config import load_config
        from TTS.tts.models import setup_model as setup_tts_model

        logging.basicConfig(format='CoquiEngine: %(message)s', level=loglevel)

        logging.info(f"Starting CoquiEngine")


        def get_conditioning_latents(filename):
            logging.debug(f"Computing speaker latents")

            if not filename or len(filename) == 0:
                filename = "coqui_default_voice.wav"

            # verify that filename ends with .wav
            if filename.endswith(".json"):
                filename_json = filename
                filename = filename[:-5]
                filename_wav = filename + "wav"
            elif filename.endswith(".wav"):
                filename_json = filename[:-3] + "json"
                filename = filename[:-3]
                filename_wav = filename + "wav"
            else:
                filename_json = filename + ".json"
                filename_wav = filename + ".wav"

            if voices_path:
                filename_voice_wav = os.path.join(voices_path, filename_wav)
                filename_voice_json = os.path.join(voices_path, filename_json)
            else:
                filename_voice_wav = filename_wav
                filename_voice_json = filename_json

            if not os.path.exists(filename_voice_json) and not os.path.exists(filename_voice_wav):
                if len(filename) > 0:
                    logging.info(f"Using default female voice, both {filename_voice_json} and {filename_voice_wav} not found.")
                else:
                    logging.info(f"Using default female voice, no cloning source specified.")
                
                # Get the directory of the current script
                current_dir = os.path.dirname(os.path.realpath(__file__))
                filename_voice_json = os.path.join(current_dir, "coqui_default_voice.json")
                if not os.path.exists(filename_voice_json):
                    raise ValueError(f"Default voice file {filename_voice_json} not found.")                

            # check if latents are already computed
            if os.path.exists(filename_voice_json):
                logging.debug(f"Latents already computed, reading from {filename_voice_json}")
                with open(filename_voice_json, "r") as new_file:
                    latents = json.load(new_file)

                speaker_embedding = (torch.tensor(latents["speaker_embedding"]).unsqueeze(0).unsqueeze(-1))
                gpt_cond_latent = (torch.tensor(latents["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0))                

                return gpt_cond_latent, speaker_embedding

            # compute and write latents to json file
            logging.debug(f"Computing latents for {filename}")

            gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(audio_path=filename_voice_wav, gpt_cond_len=30, max_ref_length=60)

            latents = {
                "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
                "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
            }
            with open(filename_voice_json, "w") as new_file:
                json.dump(latents, new_file)

            return gpt_cond_latent, speaker_embedding

        def postprocess_wave(chunk):
            """Post process the output waveform"""
            if isinstance(chunk, list):
                chunk = torch.cat(chunk, dim=0)
            chunk = chunk.clone().detach().cpu().numpy()
            chunk = chunk[None, : int(chunk.shape[0])]
            chunk = np.clip(chunk, -1, 1)
            chunk = chunk.astype(np.float32)
            return chunk

        logging.debug(f"Initializing coqui model {model_name} with cloning reference {cloning_reference_wav} and language {language}")

        try:
            logging.debug(f"Setting torch threads to {thread_count}")
            torch.set_num_threads(int(str(thread_count)))

            # Check if CUDA or MPS is available, else use CPU
            logging.debug (f"Checking for CUDA and MPS availability")
            if torch.cuda.is_available():
                logging.info("CUDA available, GPU inference used.")
                device = torch.device("cuda")
            elif use_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logging.info("MPS available, GPU inference used.")
                device = torch.device("mps")
            else:
                logging.info("CUDA and MPS not available, CPU inference used.")
                device = torch.device("cpu")

            logging.debug (f"Torch device set.")

            if local_model_path:
                logging.debug (f"Starting TTS with local path from {local_model_path} ")

                config = load_config((os.path.join(local_model_path, "config.json")))
                tts = setup_tts_model(config)
                tts.load_checkpoint(
                    config,
                    checkpoint_path=os.path.join(local_model_path, "model.pth"),
                    vocab_path=os.path.join(local_model_path, "vocab.json"),
                    eval=True,
                    use_deepspeed=use_deepspeed,
                )                
            else:
                model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
                logging.debug (f"Starting TTS with autoupdate from {model_path} ")

                config = load_config((os.path.join(model_path, "config.json")))
                tts = setup_tts_model(config)
                tts.load_checkpoint(
                    config,
                    checkpoint_path=os.path.join(model_path, "model.pth"),
                    vocab_path=os.path.join(model_path, "vocab.json"),
                    eval=True,
                    use_deepspeed=use_deepspeed,
                )
            tts.to(device)

            logging.debug("XTTS Model loaded.")


            gpt_cond_latent, speaker_embedding = get_conditioning_latents(cloning_reference_wav)

        except Exception as e:
            logging.exception(f"Error initializing main coqui engine model: {e}")
            raise

        ready_event.set()

        logging.info('Coqui text to speech synthesize model initialized successfully')

        try:
            while True:
                message = conn.recv()  
                command = message['command']
                data = message['data']

                if command == 'update_reference':
                    new_wav_path = data['cloning_reference_wav']
                    gpt_cond_latent, speaker_embedding = get_conditioning_latents(new_wav_path)
                    conn.send(('success', 'Reference updated successfully'))

                if command == 'shutdown':
                    logging.info('Shutdown command received. Exiting worker process.')
                    conn.send(('shutdown', 'shutdown'))
                    break  # This exits the loop, effectively stopping the worker process.

                elif command == 'synthesize':

                    text = data['text']
                    language = data['language']

                    logging.debug(f'Starting inference for text: {text}')

                    chunks =  tts.inference_stream(
                        text,
                        language,
                        gpt_cond_latent,
                        speaker_embedding,
                        stream_chunk_size=stream_chunk_size,
                        overlap_wav_len=overlap_wav_len,
                        temperature=temperature,
                        length_penalty=length_penalty,
                        repetition_penalty=repetition_penalty,
                        top_k=top_k,
                        top_p=top_p,
                        speed=speed,
                        enable_text_splitting=enable_text_splitting
                    )

                    if full_sentences:
                        chunklist = []

                        for i, chunk in enumerate(chunks):
                            chunk = postprocess_wave(chunk)
                            chunklist.append(chunk.tobytes())

                        for chunk in chunklist:
                            conn.send(('success', chunk))
                    else:
                        for i, chunk in enumerate(chunks):
                            chunk = postprocess_wave(chunk)
                            conn.send(('success', chunk.tobytes()))                            

                    # Send silent audio
                    sample_rate = config.audio.sample_rate  

                    if text[-1] in [","]:
                        silence_duration = 0.2  # add 200ms speaking pause in case of comma
                    else:
                        silence_duration = 0.5  # add 500ms speaking pause in case of sentence end

                    silent_samples = int(sample_rate * silence_duration)
                    silent_chunk = np.zeros(silent_samples, dtype=np.float32)
                    conn.send(('success', silent_chunk.tobytes()))                        

                    conn.send(('finished', ''))
        
        except KeyboardInterrupt:
            logging.info('Keyboard interrupt received. Exiting worker process.')
            conn.send(('shutdown', 'shutdown'))

        except Exception as e:
            logging.error(f"General synthesis error: {e} occured trying to synthesize text {text}")

            tb_str = traceback.format_exc()
            print (f"Traceback: {tb_str}")
            print (f"Error: {e}")

            conn.send(('error', str(e)))

    def send_command(self, command, data):
        """
        Send a command to the worker process.
        """
        message = {'command': command, 'data': data}
        self.parent_synthesize_pipe.send(message)            
            
    def set_cloning_reference(self, cloning_reference_wav: str):
        """
        Send an 'update_reference' command and wait for a response.
        """
        self.send_command('update_reference', {'cloning_reference_wav': cloning_reference_wav})
        
        # Wait for the response from the worker process
        status, result = self.parent_synthesize_pipe.recv()
        if status == 'success':
            logging.info('Reference WAV updated successfully')
        else:
            logging.error(f'Error updating reference WAV: {cloning_reference_wav}')

        return status, result

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for Coqui Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paFloat32, 1, 24000
    
    def _prepare_text_for_synthesis(self, text: str):
        """
        Splits a text into sentences.

        Args:
            text (str): Text to prepare for synthesis.

        Returns:
            text (str): Prepared text.
        """

        logging.debug (f"Preparing text for synthesis: \"{text}\"")

        if self.prepare_text_for_synthesis_callback:
            return self.prepare_text_for_synthesis_callback(text)

        # A fast fix for last character, may produce weird sounds if it is with text
        text = text.strip()
        text = text.replace("</s>", "")
        text = re.sub("```.*```", "", text, flags=re.DOTALL)
        text = re.sub("`.*`", "", text, flags=re.DOTALL)
        text = re.sub("\(.*\)", "", text, flags=re.DOTALL)
        text = text.replace("```", "")
        text = text.replace("...", " ")
        text = text.replace("»", "")
        text = text.replace("«", "")
        text = re.sub(" +", " ", text)
        #text= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2",text)
        #text= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2",text)

        try:
            if len(text) > 2 and text[-1] in ["."]:
                text = text[:-1] 
            elif len(text) > 2 and text[-1] in ["!", "?", ","]:
                text = text[:-1] + " " + text[-1]
            elif len(text) > 3 and text[-2] in ["."]:
                text = text[:-2] 
            elif len(text) > 3 and text[-2] in ["!", "?", ","]:
                text = text[:-2] + " " + text[-2]
        except Exception as e:
            logging.warning (f"Error fixing sentence end punctuation: {e}, Text: \"{text}\"")        
        
        text = text.strip()

        logging.debug (f"Text after preparation: \"{text}\"")

        return text

    def synthesize(self, 
                   text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """

        with self._synthesize_lock:
            text = self._prepare_text_for_synthesis(text)

            if len(text) < 1:
                return

            data = {'text': text, 'language': self.language}
            self.send_command('synthesize', data)

            status, result = self.parent_synthesize_pipe.recv()

            while not 'finished' in status:
                if 'shutdown' in status or 'error' in status:
                    if 'error' in status:
                        logging.error(f'Error synthesizing text: {text}')
                        logging.error(f'Error: {result}')
                    return False
                self.queue.put(result)
                status, result = self.parent_synthesize_pipe.recv()

            return True
    
    @staticmethod
    def download_file(url, destination):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

    @staticmethod
    def download_model(model_name = "2.0.2", local_models_path = None):

        # Creating a unique folder for each model version
        if local_models_path and len(local_models_path) > 0:
            model_folder = os.path.join(local_models_path, f'v{model_name}')
            logging.info(f"Local models path: \"{model_folder}\"")
        else:
            model_folder = os.path.join(os.getcwd(), 'models', f'v{model_name}')
            logging.info(f"Checking for models within application directory: \"{model_folder}\"")

        os.makedirs(model_folder, exist_ok=True)

        files = {
            "config.json": f"https://huggingface.co/coqui/XTTS-v2/raw/v{model_name}/config.json",
            "model.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/v{model_name}/model.pth?download=true",
            "vocab.json": f"https://huggingface.co/coqui/XTTS-v2/raw/v{model_name}/vocab.json"
        }

        for file_name, url in files.items():
            file_path = os.path.join(model_folder, file_name)
            if not os.path.exists(file_path):
                logging.info(f"Downloading {file_name}...")
                CoquiEngine.download_file(url, file_path)
                # r = requests.get(url, allow_redirects=True)
                # with open(file_path, 'wb') as f:
                #     f.write(r.content)
                logging.info(f"{file_name} downloaded successfully.")
            else:
                logging.info(f"{file_name} already exists. Skipping download.")


        return model_folder            

    def get_voices(self):
        """
        Retrieves the installed voices available for the Coqui TTS engine.
        """
        # get all files in self.voices_path directory
        files = os.listdir(self.voices_path)

        voice_file_names = []
        for file in files:
            # remove ending .wav or .json from filename
            if file.endswith(".wav"):
                file = file[:-4]
            elif file.endswith(".json"):
                file = file[:-5]
            else:
                continue

            voice_file_names.append(file)

        return voice_file_names 
    
    def set_voice(self, voice: str):
        """
        Sets the voice to be used for speech synthesis.
        """
        self.set_cloning_reference(voice)
    
    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.

        This method should be overridden by the derived class to set the desired voice parameters.
        """
        pass

    def shutdown(self):
        """
        Shuts down the engine by terminating the process and closing the pipes.
        """
        # Send shutdown command to the worker process
        logging.info('Sending shutdown command to the worker process')
        self.send_command('shutdown', {})
        
        # Wait for the worker process to acknowledge the shutdown
        try:
            status, _ = self.parent_synthesize_pipe.recv()
            if 'shutdown' in status:
                logging.info('Worker process acknowledged shutdown')
        except EOFError:
            # Pipe was closed, meaning the process is already down
            logging.warning('Worker process pipe was closed before shutdown acknowledgement')
        
        # Close the pipe connection
        self.parent_synthesize_pipe.close()

        # Terminate the process
        logging.info('Terminating the worker process')
        self.synthesize_process.terminate()

        # Wait for the process to terminate
        self.synthesize_process.join()
        logging.info('Worker process has been terminated')