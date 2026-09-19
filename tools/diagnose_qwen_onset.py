"""Read-only native timing experiment; run with the Linux service's Python."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from RealtimeTTS import QwenCpuEngine, QwenVoice

root = Path('/home/lon/Dev/realtimetts-qwen-cpu-0.8.6/voices')
entry = next(v for v in json.loads((root/'manifest.json').read_text())['voices']
             if v['name'] == 'mira_v5_spark_de')
voice = QwenVoice(name=entry['name'], language='german', ref_audio=root/entry['files'][0])
engine = QwenCpuEngine(talker_path='/home/lon/models/qwentts-gguf/qwen-talker-0.6b-base-Q8_0.gguf',
    codec_path='/home/lon/models/qwentts-gguf/qwen-tokenizer-12hz-Q8_0.gguf',
    voice=voice, cpu_threads=7, clone_mode='speaker_only', clamp_fp16=False,
    seed=42, do_sample=False, subtalker_do_sample=False, local_files_only=True,
    startup_buffer_ms=180, onset_silence_profile='qwen3_tts_12hz_0_6b_base_q8_v1')
original = engine._backend.stream
runs = []
texts = ['Heute Morgen', 'Heute Morgen gehen wir gemeinsam durch den Park und genießen die frische Luft.']
for buffer in [180, 0]:
    engine.startup_buffer_ms = buffer
    for repeat in range(2):
        for text in texts[::1 if repeat == 0 else -1]:
            chunks = []
            started = time.perf_counter()
            def recording(**kwargs):
                for audio, rate in original(**kwargs):
                    chunks.append({'ms': (time.perf_counter()-started)*1000,
                                   'samples': len(audio), 'peak': float(np.max(np.abs(audio)))})
                    yield audio, rate
            engine._backend.stream = recording
            if not engine.synthesize(text):
                raise RuntimeError(str(engine.last_error))
            profile = engine.last_synthesis_profile
            while not engine.queue.empty():
                engine.queue.get_nowait()
            run = {'text':text, 'buffer_ms':buffer, 'profile':profile, 'raw_chunks':chunks}
            runs.append(run)
            print(json.dumps({'text':text, 'buffer_ms':buffer,
                'trim_ms':profile['leading_trimmed_ms'], 'queue_ms':profile['first_queue_ms'],
                'first_chunk_ms':profile['first_chunk_duration_ms'],
                'native':profile['native'], 'raw_chunks':chunks[:5]}, ensure_ascii=False), flush=True)
engine.shutdown()
with tempfile.NamedTemporaryFile(mode='w', prefix='qwen-onset-', suffix='.json', delete=False) as f:
    json.dump(runs, f, ensure_ascii=False, indent=2)
    print('REPORT='+f.name, flush=True)
