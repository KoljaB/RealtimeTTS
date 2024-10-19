import torch.multiprocessing as mp
from .configs.config import Config
from queue import Empty
from pathlib import Path
from tqdm import tqdm
import numpy as np
import requests
import torch
import glob
import gc
import sys
import os
import re


flag_vc = False
debug = False


def _stream_play_worker(
    stream_queue: mp.Queue,
    shutdown_event: mp.Event,
    finished_event: mp.Event,
    sample_rate: int = 48000,
):
    import pyaudio

    pyaudio_instance = pyaudio.PyAudio()
    stream = pyaudio_instance.open(
        format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True
    )
    stream.start_stream()

    try:
        while not shutdown_event.is_set():
            try:
                name, chunk_to_play = stream_queue.get(timeout=0.02)
                if name == "play":
                    stream.write(chunk_to_play)
                elif name == "finished":
                    finished_event.set()
            except Empty:
                continue
    finally:
        stream.stop_stream()
        stream.close()
        pyaudio_instance.terminate()
        if debug:
            print("Audio stream closed and PyAudio terminated.")


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


class GUIConfig:
    def __init__(self) -> None:
        self.pth_path: str = None
        self.index_path: str = None
        self.pitch: int = 0.0
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.2  # s
        self.threhold: int = -60.0
        self.crossfade_time: float = 0.08
        self.extra_time: float = 2.0
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.use_pv: bool = False
        self.rms_mix_rate: float = 0.5
        self.index_rate: float = 0.0
        self.n_cpu: int = 4.0
        self.f0method: str = "fcpe"


class RealtimeRVCBase:
    def __init__(self, pth: str = None, index: str = None) -> None:
        self.gui_config = GUIConfig()
        self.config = Config()
        self.function = "vc"
        self.rvc = None
        if pth is not None:
            self.gui_config.pth_path = pth
        if index is not None:
            self.gui_config.index_path = index

    def start_vc(self):
        from .tools.torchgate import TorchGate
        import torch
        import torchaudio.transforms as tat

        torch.cuda.empty_cache()
        import rvc.tools.rvc_for_realtime as rvc_for_realtime
        from multiprocessing import Queue

        inp_q = Queue()
        opt_q = Queue()

        self.rvc = rvc_for_realtime.RVC(
            self.gui_config.pitch,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.gui_config.n_cpu,
            inp_q,
            opt_q,
            self.config,
            self.rvc if hasattr(self, "rvc") else None,
        )
        self.gui_config.samplerate = (
            self.rvc.tgt_sr
            if self.gui_config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_time * self.gui_config.samplerate / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.gui_config.crossfade_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_time * self.gui_config.samplerate / self.zc
                )
            )
            * self.zc
        )
        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.res_buffer: torch.Tensor = torch.zeros(
            2 * self.zc, device=self.config.device, dtype=torch.float32
        )
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)

    def audio_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        """
        Core of the real-time processing
        - called for every block of audio data captured from the microphone
        - handles incoming audio data, applies transformations, and sends it
          to the output
        """
        # import time
        import librosa
        import torch
        import torch.nn.functional as F

        global flag_vc
        # start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.gui_config.threhold > -60:
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-self.block_frame :] = torch.from_numpy(indata).to(
            self.config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # input noise reduction and resampling
        if self.gui_config.I_noise_reduce and self.function == "vc":
            input_wav = self.input_wav[
                -self.sola_buffer_frame - self.block_frame - 2 * self.zc :
            ]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0))[
                0, 2 * self.zc :
            ]
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.nr_buffer[:] = input_wav[self.block_frame :]
            input_wav = torch.cat((self.res_buffer[:], input_wav[: self.block_frame]))
            self.res_buffer[:] = input_wav[-2 * self.zc :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                input_wav
            )[160:]
        else:
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav[-self.block_frame - 2 * self.zc :]
            )[160:]
        # infer
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.gui_config.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        else:
            infer_wav = self.input_wav[
                -self.crossfade_frame - self.sola_search_frame - self.block_frame :
            ].clone()
        # output noise reduction
        if (self.gui_config.O_noise_reduce and self.function == "vc") or (
            self.gui_config.I_noise_reduce and self.function == "im"
        ):
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
            rms1 = librosa.feature.rms(
                y=self.input_wav_res[
                    160 * self.skip_head : 160 * (self.skip_head + self.return_length)
                ]
                .cpu()
                .numpy(),
                frame_length=640,
                hop_length=160,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]

        def phase_vocoder(a, b, fade_out, fade_in):
            window = torch.sqrt(fade_out * fade_in)
            fa = torch.fft.rfft(a * window)
            fb = torch.fft.rfft(b * window)
            absab = torch.abs(fa) + torch.abs(fb)
            n = a.shape[0]
            if n % 2 == 0:
                absab[1:-1] *= 2
            else:
                absab[1:] *= 2
            phia = torch.angle(fa)
            phib = torch.angle(fb)
            deltaphase = phib - phia
            deltaphase = deltaphase - 2 * np.pi * torch.floor(
                deltaphase / 2 / np.pi + 0.5
            )
            w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
            t = torch.arange(n).unsqueeze(-1).to(a) / n
            result = (
                a * (fade_out**2)
                + b * (fade_in**2)
                + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
            )
            return result

        if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        if sys.platform == "darwin":
            outdata[:] = infer_wav[: self.block_frame].cpu().numpy()[:, np.newaxis]
        else:
            outdata[:] = infer_wav[: self.block_frame].repeat(2, 1).t().cpu().numpy()

    def set_pitch(self, pitch):
        self.rvc.change_key(pitch)


class RealtimeRVC:
    def __init__(
        self,
        rvc_model_path,
        stop_callback=None,
        yield_chunk_callback=None,
        sample_rate=40000,
    ) -> None:
        RVC_DOWNLOAD_LINK = (
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
        )
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.dl_model(RVC_DOWNLOAD_LINK, "hubert_base.pt", BASE_DIR / "assets/hubert")

        self.sample_rate = sample_rate
        self.rvc = None

        self.rvc_model_path = rvc_model_path
        self.stop_callback = stop_callback
        self.yield_chunk_callback = yield_chunk_callback
        self.chunk_callback_only = False
        self.is_active = True
        self.stream_play_chunk_queue = mp.Queue()
        self.shutdown_event = mp.Event()
        self.finished_event = mp.Event()
        self.started = False
        self.current_model = ""
        self.muted = False

        # start stop_worker as thread
        import threading

        self.stop_worker_thread = threading.Thread(target=self.stop_worker)
        self.stop_worker_thread.isDaemon = True
        self.stop_worker_thread.start()

    def dl_model(self, link, model_name, dir_name):
        url = f"{link}{model_name}"
        file_path = Path(dir_name) / model_name

        # Check if file already exists
        if file_path.exists():
            if debug:
                print(
                    f"File '{model_name}' already exists in '{dir_name}'. Skipping download."
                )
            return

        if debug:
            print(f"Starting to download RVC model {model_name}")

        # Create directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok=True)

        # Send a GET request to fetch the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the total file size
        total_size = int(response.headers.get("content-length", 0))

        # Open the output file and initialize tqdm progress bar
        with open(file_path, "wb") as file, tqdm(
            desc=model_name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

        print(f"Download of '{model_name}' completed.")

    def set_muted(self, muted: bool = True):
        self.muted = muted

    def start(self, model_name=None):
        if debug:
            print("Starting RVC")
        self.started = True

        if debug:
            print("Starting RVC worker process")
        self.stream_play_worker_process = mp.Process(
            target=_stream_play_worker,
            args=(
                self.stream_play_chunk_queue,
                self.shutdown_event,
                self.finished_event,
                self.sample_rate,
            ),
        )
        self.stream_play_worker_process.start()

        if not model_name:
            model_name = "Samantha"

        pth_path = os.path.join(self.rvc_model_path, model_name + ".pth")
        index_path = os.path.join(self.rvc_model_path, model_name + ".index")
        if debug:
            print("Starting RVC with model", model_name)
            print("pth_path:", pth_path)
            print("index_path:", index_path)

        self.rvc = RealtimeRVCBase(pth=pth_path, index=index_path)
        self.rvc.start_vc()
        self.init()

    def set_pitch(self, pitch):
        self.rvc.set_pitch(pitch)

    def stop_worker(self):
        import time

        while self.is_active:
            if self.finished_event.is_set():
                if self.stop_callback:
                    self.stop_callback()
                self.finished_event.clear()
            time.sleep(0.02)

    def init(self):
        self.accumulated_chunk = []
        self.accumulated_length = 0

    def feed(self, audio_chunk, samplerate=24000, targetrate=None):
        # print("i", end="", flush=True)
        import librosa

        if targetrate is None:
            targetrate = self.sample_rate

        # Step 1: Convert to float32
        audio_chunk = (
            np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Step 2: Resample from original sample rate to 40000 Hz
        audio_chunk = librosa.resample(
            audio_chunk, orig_sr=samplerate, target_sr=targetrate
        )

        # Step 3: Accumulate chunks
        self.accumulated_chunk += audio_chunk.tolist()
        self.accumulated_length += len(audio_chunk)

        # Check if accumulated chunk reached the required block size
        if self.accumulated_length >= self.rvc.block_frame:
            # Step 4: Prepare np.ndarray
            processed_chunk = np.array(self.accumulated_chunk[: self.rvc.block_frame])

            # Reset the accumulated chunk
            self.accumulated_chunk = self.accumulated_chunk[self.rvc.block_frame :]
            self.accumulated_length -= self.rvc.block_frame

            # Step 5: Process the chunk
            outdata = np.zeros((self.rvc.block_frame, 2), dtype="float32")

            self.rvc.audio_callback(
                processed_chunk, outdata, self.rvc.block_frame, None, None
            )

            outdata_list = outdata[:, 0].tolist()
            write_data = np.array(outdata_list, dtype="float32").tobytes()

            if not self.muted:
                if not self.chunk_callback_only:
                    self.stream_play_chunk_queue.put(("play", write_data))
                else:
                    if self.yield_chunk_callback:
                        self.yield_chunk_callback(write_data)

    def is_playing(self):
        return self.stream_play_chunk_queue.qsize() > 0

    def feed_finished(self):
        self.stream_play_chunk_queue.put(("finished", None))

    def clear_queue(self):
        while self.stream_play_chunk_queue.qsize() > 0:
            try:
                self.stream_play_chunk_queue.get()
            except Empty:
                break

    def stop(self):
        self.clear_queue()

        if self.stop_callback:
            self.stop_callback()

    def shutdown(self):
        if debug:
            print("Shutting down RealtimeRVC...")
        self.is_active = False
        self.shutdown_event.set()

        # Clear the queue
        self.clear_queue()

        # Stop the worker process
        if hasattr(self, "stream_play_worker_process"):
            self.stream_play_worker_process.join(timeout=5)
            if self.stream_play_worker_process.is_alive():
                self.stream_play_worker_process.terminate()

        # Unload the model
        self.unload_model()

        # Wait for the stop worker thread to finish
        self.stop_worker_thread.join(timeout=5)

        if debug:
            print("RealtimeRVC shutdown complete.")

    def get_models(self):
        models = []
        for file in glob.glob(os.path.join(self.rvc_model_path, "*.pth")):
            model_name = os.path.basename(file)
            model_name = re.sub(r"\.pth$", "", model_name)

            # check if corresponding .index file exists
            index_file = os.path.join(self.rvc_model_path, model_name + ".index")
            if os.path.exists(index_file):
                models.append(model_name)

        return models

    def unload_model(self):
        if self.rvc:
            del self.rvc
            torch.cuda.empty_cache()
            gc.collect()
            self.rvc = None

    def set_model(self, model_name):
        if self.current_model == model_name:
            return
        self.current_model = model_name
        self.unload_model()
        self.rvc = RealtimeRVCBase(
            pth=os.path.join(self.rvc_model_path, model_name + ".pth"),
            index=os.path.join(self.rvc_model_path, model_name + ".index"),
        )
        self.rvc.start_vc()
