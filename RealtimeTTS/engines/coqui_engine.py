from multiprocessing import Process, Pipe, Event
from .base_engine import BaseEngine
import numpy as np
import traceback
import logging
import pyaudio
import torch
import json
import os
import re


class CoquiEngine(BaseEngine):

    def __init__(self, 
                 model_name = "tts_models/multilingual/multi-dataset/xtts_v2",
                 cloning_reference_wav: str = "female.wav",
                 language = "en",
                 speed = 1.0,
                 thread_count = 24,
                 stream_chunk_size = 20,
                 full_sentences = False,
                 level=logging.WARNING
                 ):
        """
        Initializes a coqui voice realtime text to speech engine object.

        Args:
            model_name (str): Name of the coqui model to use. Tested with XTTS only.
            cloning_reference_wav (str): Name to the file containing the voice to clone. Works with a 44100Hz or 22050Hz mono 32bit float WAV file.
            language (str): Language to use for the coqui model.
        """

        self.model_name = model_name
        self.language = language
        self.cloning_reference_wav = cloning_reference_wav
        self.speed = speed

        # Start the worker process
        self.main_synthesize_ready_event = Event()
        self.parent_synthesize_pipe, child_synthesize_pipe = Pipe()

        # download coqui model
        from TTS.utils.manage import ModelManager
        logging.info(f"Downloading XTTS Model: {model_name}")
        ModelManager().download_model(model_name)

        self.synthesize_process = Process(target=CoquiEngine._synthesize_worker, args=(child_synthesize_pipe, model_name, cloning_reference_wav, language, self.main_synthesize_ready_event, level, self.speed, thread_count, stream_chunk_size, full_sentences))
        self.synthesize_process.start()

        logging.debug('Waiting for coqui text to speech synthesize model to start')
        self.main_synthesize_ready_event.wait()
        logging.info('Coqui synthesis model ready')


    @staticmethod
    def _synthesize_worker(conn, model_name, cloning_reference_wav, language, ready_event, loglevel, speed, thread_count, stream_chunk_size, full_sentences):
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

            # verify that filename ends with .wav
            if not filename.endswith(".wav"):
                raise ValueError(f"Filename {filename} must end with .wav")

            filename_json = filename[:-3] + "json"

            # check if latents are already computed
            if os.path.exists(filename_json):
                logging.debug(f"Latents already computed, reading from {filename_json}")
                with open(filename_json, "r") as new_file:
                    latents = json.load(new_file)

                speaker_embedding = (torch.tensor(latents["speaker_embedding"]).unsqueeze(0).unsqueeze(-1))
                gpt_cond_latent = (torch.tensor(latents["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0))                

                return gpt_cond_latent, speaker_embedding

            # compute and write latents to json file
            logging.debug(f"Computing latents for {filename}")

            gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(audio_path=filename, gpt_cond_len=30, max_ref_length=60)

            latents = {
                "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
                "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
            }
            with open(filename_json, "w") as new_file:
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

        logging.debug(f"Initializing coqui model {model_name} with cloning reference wave file {cloning_reference_wav} and language {language}")

        try:
            torch.set_num_threads(int(str(thread_count)))

            # Check if CUDA or MPS is available, else use CPU
            if torch.cuda.is_available():
                logging.info("CUDA available, GPU inference used.")
                device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logging.info("MPS available, GPU inference used.")
                device = torch.device("mps")
            else:
                logging.info("CUDA and MPS not available, CPU inference used.")
                device = torch.device("cpu")

            model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

            print (model_path)

            config = load_config((os.path.join(model_path, "config.json")))
            tts = setup_tts_model(config)
            tts.load_checkpoint(
                config,
                checkpoint_path=os.path.join(model_path, "model.pth"),
                vocab_path=os.path.join(model_path, "vocab.json"),
                eval=True,
                use_deepspeed=False,
            )
            tts.to(device)
            logging.debug("XTTS Model loaded.")

            gpt_cond_latent, speaker_embedding = get_conditioning_latents(cloning_reference_wav)

        except Exception as e:
            logging.exception(f"Error initializing main faster_whisper transcription model: {e}")
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
                    conn.send({'status': 'shutdown'})
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
                        # decoder = "ne_hifigan",
                        stream_chunk_size=stream_chunk_size,
                        # repetition_penalty=10.0,
                        # temperature=0.75,
                        repetition_penalty=7.0,
                        temperature=0.85,
                        speed=speed,
                        enable_text_splitting=True
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
            conn.send({'status': 'shutdown'})

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
        Returns the audio stream configuration information suitable for PyAudio.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paFloat32, 1, 24000

    def synthesize(self, 
                   text: str):
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """

        # A fast fix for last chacter, may produce weird sounds if it is with text
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
            if text[-1] in ["."]:
                text = text[:-1] 
            elif text[-1] in ["!", "?", ","]:
                text = text[:-1] + " " + text[-1]
            elif text[-2] in ["."]:
                text = text[:-2] 
            elif text[-2] in ["!", "?", ","]:
                text = text[:-2] + " " + text[-2]
        except Exception as e:
            logging.warning (f"Error fixing sentence end punctuation: {e}, Text: \"{text}\"")

        text = text.strip()

        if len(text) <= 3:
            return

        data = {'text': text, 'language': self.language}
        self.send_command('synthesize', data)

        status, result = self.parent_synthesize_pipe.recv()

        while not 'finished' in status:
            if 'shutdown' in status:
                break
            if not 'error' in status:
                self.queue.put(result)
            status, result = self.parent_synthesize_pipe.recv()         

    def get_voices(self):
        """
        Retrieves the voices available in the underlying system's speech engine.
        """        
        return []
    
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
            response = self.parent_synthesize_pipe.recv()
            if response.get('status') == 'shutdown':
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