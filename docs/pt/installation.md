> **Nota:** A instalação básica com `pip install realtimetts` não é mais recomendada, use `pip install realtimetts[all]` em vez disso.

A biblioteca RealtimeTTS oferece opções de instalação para várias dependências de acordo com o seu caso de uso. Aqui estão as diferentes maneiras de instalar o RealtimeTTS dependendo das suas necessidades:

### Instalação Completa

Para instalar o RealtimeTTS com suporte para todos os motores TTS:

```
pip install -U realtimetts[all]
```

### Instalação Personalizada

RealtimeTTS permite uma instalação personalizada com instalações mínimas de bibliotecas. Aqui estão as opções disponíveis:
- **todos**: Instalação completa com todos os motores suportados.
- **sistema**: Inclui capacidades de TTS específicas do sistema (e.g., pyttsx3).
- **azure**: Adiciona suporte ao Azure Cognitive Services Speech.
- **elevenlabs**: Inclui integração com a API da ElevenLabs.
- **openai**: Para serviços de voz da OpenAI.
- **gtts**: Suporte ao Google Text-to-Speech.
- **coqui**: Instala o mecanismo Coqui TTS.
- **mínimo**: Instala apenas os requisitos básicos sem motor (only needed if you want to develop an own engine)


Digamos que você quer instalar o RealtimeTTS apenas para uso local do Coqui TTS neuronal, então você deve usar:

```
pip install realtimetts[coqui]
```

Por exemplo, se você quiser instalar o RealtimeTTS com suporte apenas para Azure Cognitive Services Speech, ElevenLabs e OpenAI:

``` 
pip install realtimetts[azure,elevenlabs,openai]
```

### Instalação em Ambiente Virtual

Para aqueles que desejam realizar uma instalação completa dentro de um ambiente virtual, sigam estes passos:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

Mais informações sobre [instalação do CUDA](#cuda-installation).

## Requisitos do Motor

Diferentes motores suportados pelo RealtimeTTS têm requisitos únicos. Certifique-se de cumprir esses requisitos com base no motor que você escolher.

### SystemEngine
O `SystemEngine` funciona imediatamente com as capacidades de TTS integradas do seu sistema. Nenhuma configuração adicional é necessária.

### GTTSEngine
O `GTTSEngine` funciona imediatamente usando a API de texto para fala do Google Translate. Nenhuma configuração adicional é necessária.

### OpenAIEngine
Para usar o `OpenAIEngine`:
- defina a variável de ambiente OPENAI_API_KEY
- instale o ffmpeg (veja o ponto 3 da [instalação do CUDA](#cuda-installation))

### AzureEngine
Para usar o `AzureEngine`, você precisará:
- Chave da API do Microsoft Azure Text-to-Speech (fornecida através do parâmetro do construtor do AzureEngine "speech_key" ou na variável de ambiente AZURE_SPEECH_KEY)
- Região do serviço Microsoft Azure.

Certifique-se de que você tenha essas credenciais disponíveis e corretamente configuradas ao inicializar o `AzureEngine`.

### ElevenlabsEngine
Para o `ElevenlabsEngine`, você precisa:
- Chave de API do Elevenlabs (fornecida através do parâmetro "api_key" do construtor do ElevenlabsEngine ou na variável de ambiente ELEVENLABS_API_KEY)
- `mpv` instalado no seu sistema (essential for streaming mpeg audio, Elevenlabs only delivers mpeg).

  🔹 **Instalando `mpv`:**
  - **macOS**:
    ``` 
    brew install mpv
```

  - **Linux e Windows**: Visite [mpv.io](https://mpv.io/) para instruções de instalação.

### CoquiEngine

Entrega TTS neural de alta qualidade, local, com clonagem de voz.

Baixa primeiro um modelo de TTS neural. Na maioria dos casos, será rápido o suficiente para Realtime usando síntese por GPU. Precisa de cerca de 4-5 GB de VRAM.

- para clonar uma voz, envie o nome do arquivo de um arquivo WAV contendo a voz de origem como parâmetro "voice" para o construtor do CoquiEngine
- a clonagem de voz funciona melhor com um arquivo WAV mono de 16 bits a 22050 Hz contendo uma amostra curta (~5-30 seg)

Na maioria dos sistemas, será necessário suporte de GPU para rodar rápido o suficiente para o tempo real, caso contrário, você experimentará gagueira.

### Instalação do CUDA

Esses passos são recomendados para aqueles que necessitam de **melhor desempenho** e possuem uma GPU NVIDIA compatível.

> **Nota**: *para verificar se sua GPU NVIDIA suporta CUDA, visite a [lista oficial de GPUs CUDA](https://developer.nvidia.com/cuda-gpus).*

Para usar uma torch com suporte via CUDA, siga estes passos:

> **Nota**: *instalações mais recentes do pytorch [podem](https://stackoverflow.com/a/77069523) (não verificado) não precisar mais da instalação do Toolkit (e possivelmente do cuDNN).*

1. **Instale o NVIDIA CUDA Toolkit**:
   Por exemplo, para instalar o Toolkit 12.X, por favor
   - Visite [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
    - Selecione seu sistema operacional, arquitetura do sistema e versão do sistema operacional.
    - Baixe e instale o software.

    ou para instalar o Toolkit 11.8, por favor
    - Visite [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Selecione seu sistema operacional, arquitetura do sistema e versão do sistema operacional.
    - Baixe e instale o software.

2. **Instale o NVIDIA cuDNN**:

    Por exemplo, para instalar o cuDNN 8.7.0 para CUDA 11.x, por favor
    - Visite [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Clique em "Download cuDNN v8.7.0 (28 de novembro de 2022), para CUDA 11.x".
    - Baixe e instale o software.

3. **Instale o ffmpeg**:

    Você pode baixar um instalador para o seu sistema operacional no [Site do ffmpeg](https://ffmpeg.org/download.html).

    Ou use um gerenciador de pacotes:

    - **No Ubuntu ou Debian**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **No Arch Linux**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **No MacOS usando Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ``` 
        brew install ffmpeg
        ```

    - **No Windows usando Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ``` 
        choco install ffmpeg
        ```

    - **No Windows usando Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        scoop install ffmpeg
        ```

4. **Instale o PyTorch com suporte a CUDA**:

    Para atualizar sua instalação do PyTorch e habilitar o suporte a GPU com CUDA, siga estas instruções com base na sua versão específica do CUDA. Isso é útil se você deseja melhorar o desempenho do RealtimeSTT com as capacidades do CUDA.

    - **Para CUDA 11.8:**

        Para atualizar o PyTorch e o Torchaudio para suportar CUDA 11.8, use os seguintes comandos:

        ``` 
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 
```
        ```

    - **Para CUDA 12.X:**


        Para atualizar o PyTorch e o Torchaudio para suportar CUDA 12.X, execute o seguinte:

        ``` 
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 
```
        ```

    Substitua `2.3.1` pela versão do PyTorch que corresponda ao seu sistema e requisitos.

5. **Correção para resolver problemas de compatibilidade**:
    Se você encontrar problemas de compatibilidade com bibliotecas, tente definir essas bibliotecas para versões fixas:

  ``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
```