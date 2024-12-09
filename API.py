import os
import re
import tempfile
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

import sys
sys.path.append(f"../../{os.path.dirname(os.path.abspath(__file__))}/third_party/BigVGAN/")

import tqdm
import matplotlib
import matplotlib.pylab as plt
import hashlib

from huggingface_hub import snapshot_download
from importlib.resources import files
from pathlib import Path
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)
from flask import Flask, request, jsonify, send_file

# Inicializar Flask
app = Flask(__name__)

# Crear la carpeta de salida si no existe
os.makedirs("Output", exist_ok=True)

matplotlib.use("Agg")


try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# -----------------------------------------

#Decorador para habilitar la ejecución en GPU si 'spaces' está disponible
def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
)

#Carga y configura un modelo de síntesis de voz con sus parámetros y vocabulario
def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("tokenizer : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


# load vocoder
#Carga un vocoder especificado desde un camino local o descargándolo de Hugging Face
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device):
    if vocoder_name == "vocos":
        if is_local:
            print(f"Load vocos from local path {local_path}")
            vocoder = Vocos.from_hparams(f"{local_path}/config.yaml")
            state_dict = torch.load(f"{local_path}/pytorch_model.bin", map_location="cpu")
            vocoder.load_state_dict(state_dict)
            vocoder = vocoder.eval().to(device)
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder

vocoder = load_vocoder()


#Inicializa el pipeline de reconocimiento de voz automático con el modelo Whisper de OpenAI
def initialize_asr_pipeline(device=device):
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16,
        device=device,
    )

# Carga un checkpoint en el modelo especificado desde la ruta dada.
def load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16 if device == "cuda" and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device)

# load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
)

# Realiza la inferencia generando audio a partir de texto y audio de referencia.
def infer(ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info):

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    #if model == "F5-TTS":
    ema_model = F5TTS_ema_model
    #elif model == "E2-TTS":
    #    ema_model = E2TTS_ema_model
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remueve Silencios
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave)

# Divide el texto en oraciones según la puntuación seguida de un espacio
def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    prediction_model=None,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return infer_batch_process(
        (audio, sr),
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        prediction_model=prediction_model,
        mel_spec_type=mel_spec_type,
        progress=progress,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
    )


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    prediction_model=None,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=0.8,
    fix_duration=None,
    device=None,
):
    
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    for i, gen_text in enumerate(progress(gen_text_batches)):
        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        duration=None
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        elif prediction_model:
            duration_in_sec = prediction_model(audio, text_list)
            print(duration_in_sec)
            frame_rate = model_obj.mel_spec.target_sample_rate // model_obj.mel_spec.hop_length
            duration = (duration_in_sec * frame_rate / speed).to(torch.long).item()
            print(f"Got duration of {duration} frames ({duration_in_sec.item()} secs) for generated speech")

        if duration is None:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated_mel_spec)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    return final_wave, target_sample_rate, combined_spectrogram


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")

def generate_wav_file_name():
    """
    Genera un nombre de archivo .wav basado en la fecha y hora actual.
    
    Returns:
        str: El nombre de archivo .wav generado.
    """
    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)
    
    file_name = f"{year}-{month}-{day}_{hour}-{minute}-{second}.wav"
    return file_name


# Ruta para generar audio
@app.route('/generate', methods=['POST'])
def generate_audio():
    try:
        # Lógica para manejar la generación de audio
        data = request.get_json()
        text = data.get("text", "")
        codec = data.get("codec", "wav")

        if not text:
            return jsonify({"error": "El campo 'text' es obligatorio."}), 400
        if codec.lower() != "wav":
            return jsonify({"error": "Solo se admite el formato 'wav'."}), 400

        # Generar audio
        audio_result, _ = infer(
            ref_audio_orig="Ref//Audio-demo.mp3",  # Ajustar si es necesario
            ref_text="",
            gen_text=text,
            model="",
            remove_silence=0,
            cross_fade_duration=0.15,
            speed=1.0,
            show_info=print,
        )

        # Guardar el audio generado
        audio_filename = os.path.join("Output", generate_wav_file_name())
        torchaudio.save(audio_filename, torch.from_numpy(_).unsqueeze(0), target_sample_rate)

        # Enviar el archivo generado
        return send_file(audio_filename, as_attachment=True, download_name=generate_wav_file_name())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta para la raíz
@app.route('/')
def home():
    return jsonify({"message": "API para el TTS"})



# Ejecutar el servidor
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
