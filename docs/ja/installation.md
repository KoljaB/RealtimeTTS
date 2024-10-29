> **注意:** `pip install realtimetts` での基本インストールは推奨されなくなりました。代わりに `pip install realtimetts[all]` を使用してください。

RealtimeTTSライブラリは、あなたのユースケースに応じたさまざまな依存関係のインストールオプションを提供します。 ニーズに応じてRealtimeTTSをインストールする方法はいくつかあります:

### フルインストール

すべてのTTSエンジンに対応したRealtimeTTSをインストールするには:

```
pip install -U realtimetts[all]
```

### カスタムインストール

RealtimeTTSは、最小限のライブラリインストールでカスタムインストールを可能にします。 こちらが利用可能なオプションです。
- **すべて**: すべてのエンジンがサポートされた完全なインストール。
- **システム**: システム固有のTTS機能を含む (e.g., pyttsx3).
- **azure**: Azure Cognitive Services Speechサポートを追加します。
- **elevenlabs**: ElevenLabs APIとの統合が含まれています。
- **openai**: OpenAIの音声サービス用。
- **gtts**: Google テキスト読み上げサポート。
- **coqui**: Coqui TTSエンジンをインストールします。
- **minimal**: エンジンなしで基本要件のみをインストール (only needed if you want to develop an own engine)


ローカルの神経系Coqui TTSでのみRealtimeTTSをインストールしたい場合は、次のようにしてください:

```
pip install realtimetts[coqui]
```

例えば、Azure Cognitive Services Speech、ElevenLabs、OpenAIのサポートのみでRealtimeTTSをインストールしたい場合：

```
pip install realtimetts[azure,elevenlabs,openai]
```

### 仮想環境のインストール

仮想環境内で完全なインストールを行いたい方は、以下の手順に従ってください。

```plaintext
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.
```バット
python.exe -m pip install --pipをアップグレード
pip install -U realtimetts[all]
テキストを翻訳してください: ```

[CUDAインストール](#cuda-installation)に関する詳細情報。

## エンジン要件

RealtimeTTSがサポートする異なるエンジンには、それぞれ独自の要件があります。 選択したエンジンに基づいてこれらの要件を満たすようにしてください。

### システムエンジン
`SystemEngine`は、システムに内蔵されたTTS機能とそのまま使えます。 追加の設定は必要ありません。

### GTTSEngine
`GTTSEngine`は、Google翻訳のテキスト読み上げAPIを使用して、すぐに使える状態で動作します。 追加の設定は必要ありません。

### OpenAIエンジン
`OpenAIEngine`を使用するには:
- 環境変数 OPENAI_API_KEY を設定する
- ffmpeg をインストールする（[CUDA インストール](#cuda-installation)の3番目のポイントを参照）

### AzureEngine
`AzureEngine`を使用するには、次のものが必要です：
- Microsoft Azure Text-to-Speech APIキー（AzureEngineコンストラクタのパラメータ「speech_key」または環境変数AZURE_SPEECH_KEYで提供されます）
- マイクロソフトAzureサービスリージョン。

`AzureEngine`を初期化する際に、これらの資格情報が利用可能で正しく設定されていることを確認してください。

### ElevenlabsEngine
`ElevenlabsEngine`には、次のものが必要です：
- Elevenlabs APIキー（ElevenlabsEngineコンストラクタのパラメータ「api_key」または環境変数ELEVENLABS_API_KEYで提供されます）
- システムに`mpv`がインストールされています (essential for streaming mpeg audio, Elevenlabs only delivers mpeg).

  🔹 **`mpv`のインストール:**
  - **macOS**:
    ```plaintext
brew install mpv
```

  - **LinuxおよびWindows**: インストール手順については[mpv.io](https://mpv.io/)をご覧ください。

### CoquiEngine

高品質でローカルなニューラルTTSをボイスクローン機能付きで提供します。

まず、ニューラルTTSモデルをダウンロードします。 ほとんどの場合、GPU合成を使用すればリアルタイムで十分な速度になります。 約4〜5GBのVRAMが必要です。

- 声をクローンするには、ソース音声を含むWAVファイルのファイル名を「voice」パラメータとしてCoquiEngineコンストラクタに提出してください。
- 声のクローンは、短い（約5〜30秒）サンプルを含む22050 Hzモノ16ビットWAVファイルで最も効果的です。

ほとんどのシステムでは、リアルタイムで十分な速度を出すためにGPUサポートが必要です。さもなければ、カクつきが発生します。

### CUDAのインストール

これらの手順は、**より良いパフォーマンス**を必要とし、互換性のあるNVIDIA GPUを持っている方に推奨されます。

> **注意**: *お使いのNVIDIA GPUがCUDAをサポートしているか確認するには、[公式CUDA GPUリスト](https://developer.nvidia.com/cuda-gpus)をご覧ください。*

CUDAをサポートするTorchを使用するには、次の手順に従ってください。

> **注意**: *新しいPyTorchのインストールでは、[かもしれない](https://stackoverflow.com/a/77069523)（未確認）がToolkit（およびおそらくcuDNN）のインストールを必要としないかもしれません。*

1. **NVIDIA CUDA Toolkitをインストールする**:
    例えば、Toolkit 12.Xをインストールするには、
    - [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)を訪れてください。
    - オペレーティングシステム、システムアーキテクチャ、およびOSバージョンを選択してください。
    ソフトウェアをダウンロードしてインストールしてください。

    または、Toolkit 11.8をインストールするには、
    - [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)をご覧ください。
    - オペレーティングシステム、システムアーキテクチャ、およびOSバージョンを選択してください。
    ソフトウェアをダウンロードしてインストールしてください。

2. **NVIDIA cuDNNをインストールする**:

    例えば、CUDA 11.x用のcuDNN 8.7.0をインストールするには、次の手順を行ってください。
    - [NVIDIA cuDNNアーカイブ](https://developer.nvidia.com/rdp/cudnn-archive)を訪問してください。
    「Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x」をクリックしてください。
    ソフトウェアをダウンロードしてインストールしてください。

3. **ffmpegをインストールする**:

    [ffmpegウェブサイト](https://ffmpeg.org/download.html)からお使いのOS用のインストーラーをダウンロードできます。

    またはパッケージマネージャーを使用してください:

    - **UbuntuまたはDebianの場合**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **Arch Linuxで**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **Homebrewを使用してMacOSで** ([https://brew.sh/](https://brew.sh/)):
        ```plaintext
brew install ffmpeg
```

    - **Chocolateyを使用してWindowsで** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
```

    - **Scoopを使用してWindowsで** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        scoop install ffmpeg
```

4. **CUDAサポート付きのPyTorchをインストールする**:

    CUDAでGPUサポートを有効にするためにPyTorchのインストールをアップグレードするには、特定のCUDAバージョンに基づいてこれらの指示に従ってください。 これは、CUDA機能を使用してRealtimeSTTのパフォーマンスを向上させたい場合に役立ちます。

    - **CUDA 11.8の場合：**

        PyTorchとTorchaudioをCUDA 11.8に対応させるために、次のコマンドを使用してください。

        ```plaintext
pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
        テキストを翻訳する: ```

    - **CUDA 12.Xの場合：**


        PyTorchとTorchaudioをCUDA 12.Xに対応させるために、次のコマンドを実行してください。

        ```plaintext
pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
        テキストを翻訳する: ```

    `2.3.1` をあなたのシステムと要件に合ったPyTorchのバージョンに置き換えてください。

5. **互換性の問題を解決するための修正**:
    ライブラリの互換性の問題が発生した場合は、これらのライブラリを固定バージョンに設定してみてください。

  テキストを翻訳する: ``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```