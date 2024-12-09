# Detalles del Proyecto
Este proyecto consta de un TTS (Text To Speech) para la empresa Alloxentric, en el cual a través de una API, se envía el texto que se desea generar a voz y este devuelve un archivo .wav con el audio generado por un modelo de F5-TTS.

## Prerequisites
- Python 3.11 - https://www.python.org/downloads/release/python-3119/
- git - https://git-scm.com/downloads/win
- ffmpeg: https://www.ffmpeg.org/download.html#build-windows
  - Watch a tutorial here: https://www.youtube.com/watch?v=JR36oH35Fgg&t=159s&ab_channel=Koolac
- Nvidia GPU with more than 8gb of VRAM 

# Instalacion
## Seleccionar la ubicación del directorio
Se recomienda elegir una ubicación fácilmente accesible. Para seleccionar la ubicación, utilice el siguiente comando:
```bash
cd "Inserte aqui su ubicacion"
```
## Descargar los archivos al directorio 

Se debe realizar una copia del repositorio ubicado en GitHub utilizando el siguiente comando:

```bash
git clone https://github.com/Carlos-not-found/TTS-2.0
```

El directorio debe contener una serie de carpetas y archivos, siendo los más importantes los siguientes:
main.py: Archivo de la API del servicio. Para utilizar la versión original del programa, diríjase a src\f5_tts\infer\infer_gradio.py.
request.py: Archivo que contiene la solicitud correspondiente y una señal de respuesta correcta o incorrecta de la API.

Se proporcionará una explicación más extensa en los pasos posteriores.

## Instalación de los requerimientos básicos
En este paso, una vez dentro de la carpeta, se debe ejecutar el siguiente comando para construir el ambiente virtual:

```bash
py -3.11 -m venv venv
```


"Una vez creado el ambiente virtual, se procederá a iniciar el entorno de trabajo:
```bash
venv\Scripts\activate
```
Se comenzará instalando las primeras librerías utilizando el siguiente comando:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```
Finalmente, se instalarán los recursos necesarios para el programa.

```bash
pip install -e .
```

## Inicio preventivo
Para iniciar la API, primero se debe ejecutar el programa original, con el objetivo de que se generen las carpetas necesarias para implementar el modelo en español. Para ello, inicie el programa con el siguiente comando:

```bash
f5-tts_infer-gradio
```
Una vez iniciado, en la terminal se debería ver una línea similar a el siguiente ejemplo 

```bash
model : C:\Users\"nombre del usuario"\.cache\huggingface\hub\models--SWivid--F5-TTS\snapshots\995ff41929c08ff968786b448a384330438b5cb6\F5TTS_Base\model_1200000.safetensors
```
A continuación se descargará un archivo con el mismo nombre al de la dirección en la siguiente página

https://huggingface.co/jpgallegoar/F5-Spanish/tree/main 

Una vez descargado se dirigirá a la dirección encontrada anteriormente, allí verá el archivo mencionado en la dirección, agregue la extensión “.bak” al archivo y mueva el descargado anteriormente a esta carpeta.

Una vez hecho esto el programa funcionará según lo necesite.

## Iniciar API
Para iniciar la API, se debe ejecutar el archivo “main.py” que se encuentra en la carpeta principal.
Después de ejecutar el archivo, se podrá acceder a la API a través de la dirección correspondiente (por defecto, localhost:5000).
Es necesario asegurarse de que todos los módulos se hayan cargado correctamente y de que no haya errores en la terminal.
Se recomienda realizar una prueba de la API haciendo una petición a http://localhost:5000 para verificar que esté funcionando correctamente.


# Instalacion segun programa original (en ingles)

### 1. Installation
```

Then you can choose from a few options below:

### 1. As a pip package (if just for inference)

```bash
pip install git+https://github.com/JarodMica/F5-TTS.git
```

### 2. Local editable (if also do training, finetuning)

```bash
git clone https://github.com/JarodMica/F5-TTS.git
cd F5-TTS
# I am using venv at py 3.11
py -3.11 -m venv venv
venv\Scripts\activate
# git submodule update --init --recursive  # (optional, if need bigvgan)
pip install -e .
```
If you initialize submodule, you should add the following code at the beginning of `src/third_party/BigVGAN/bigvgan.py`.
```python
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

### 3. Docker usage
```bash
# Build from Dockerfile
docker build -t f5tts:v1 .

# Or pull from GitHub Container Registry
docker pull ghcr.io/jarodmica/f5-tts:main
```


## Inference

### 1. Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct

```bash
# Launch a Gradio app (web interface)
f5-tts_infer-gradio

# Specify the port/host
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# Launch a share link
f5-tts_infer-gradio --share
```

### 2. CLI Inference

```bash
# Run with flags
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "ref_audio.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."

# Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# Or with your own .toml file
f5-tts_infer-cli -c custom.toml

# Multi voice. See src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```

### 3. More instructions

- In order to have better generation results, take a moment to read [detailed guidance](src/f5_tts/infer).
- The [Issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue) in the main repo are very useful, please try to find the solution by properly searching the keywords of problem encountered. If no answer found, then feel free to open an issue.


## Training

### 1. Gradio App

Read [training & finetuning guidance](src/f5_tts/train) for more instructions.

```bash
# Quick start with Gradio web interface
f5-tts_finetune-gradio
```


## [Evaluation](src/f5_tts/eval)


## Development

Use pre-commit to ensure code quality (will run linters and formatters automatically)

```bash
pip install pre-commit
pre-commit install
```

When making a pull request, before each commit, run: 

```bash
pre-commit run --all-files
```

Note: Some model components have linting exceptions for E722 to accommodate tensor notation


## Acknowledgements

- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation with MLX framework by [Lucas Newman](https://github.com/lucasnewman)
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) ONNX Runtime version by [DakeQQ](https://github.com/DakeQQ)

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
