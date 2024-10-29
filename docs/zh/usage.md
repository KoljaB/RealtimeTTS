# 用法

## 快速开始

这是一个基本的使用示例：

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine
```

engine = SystemEngine() # 替换为你的TTS引擎
流 = 文本转语音流(engine)
stream.feed("你好，世界！") 你今天怎么样？
stream.play_async()

## 供稿文本

你可以输入单个字符串：

```python
stream.feed("你好，这是一句话。")
```

或者你可以为实时流媒体提供生成器和字符迭代器：

```python
def write(prompt: str):
    for chunk in openai.
```聊天完成。创建(
        模型="gpt-3.5-turbo",
        消息=[{"角色": "用户", "内容": prompt}],
        流式=True
    ):
        如果 (text_chunk := chunk["choices"]上下文：[0]  
文本翻译：[0]["德尔塔"].get("content")) 不是 None:
            生成文本块

文本流 = 写入("A three-sentence relaxing speech.")

stream.feed(text_stream)

```python
char_iterator = iter("逐字符流式传输。")
stream.feed(char_iterator)
```

## 播放

异步地：

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

同步：

```python
stream.play()
```


## 测试库

测试子目录包含一组脚本，帮助您评估和理解RealtimeTTS库的功能。

请注意，大多数测试仍然依赖于“旧”的OpenAI API（<1.0.0）。 在 openai_1.0_test.py 中演示了新 OpenAI API 的用法。

- **simple_test.py**
    - **描述**：一个“你好，世界”风格的演示，展示了该库的最简单用法。

- **complex_test.py**
    - **描述**：一个全面的演示，展示了该库提供的大多数功能。

- **coqui_test.py**
    - **描述**：本地coqui TTS引擎的测试。

- **translator.py**
    - **依赖项**: 运行 `pip install openai realtimestt`。
    - **描述**：实时翻译成六种不同的语言。

- **openai_voice_interface.py**
    - **依赖项**: 运行 `pip install openai realtimestt`。
    - **描述**: 唤醒词激活并通过语音界面访问OpenAI API。

- **advanced_talk.py**
    - **依赖项**: 运行 `pip install openai keyboard realtimestt`。
    - **描述**: 在开始AI对话之前选择TTS引擎和声音。

- **minimalistic_talkbot.py**
    - **依赖项**: 运行 `pip install openai realtimestt`。
    - **描述**: 一个20行代码的基本对话机器人。

- **simple_llm_test.py**
    - **依赖项**: 运行 `pip install openai`。
    - **描述**: 如何将库与大型语言模型集成的简单演示 (LLMs).

- **test_callbacks.py**
    - **依赖项**: 运行 `pip install openai`。
    - **描述**: 展示回调并让您在实际应用环境中检查延迟时间。

## 暂停、继续和停止

暂停音频流：

```python
stream.pause()
```

恢复暂停的直播：

```python
stream.resume()
```

立即停止直播：

```python
stream.stop()
```

## 需求说明

- **Python 版本**：
  - **要求**：Python >= 3.9，< 3.13
  - **原因**： 该库依赖于来自coqui的GitHub库“TTS”，该库需要在此范围内的Python版本。

- **PyAudio**：用于创建输出音频流

- **stream2sentence**：将输入的文本流拆分成句子

- **pyttsx3**：系统文本转语音引擎

- **pydub**：用于转换音频块格式

- **azure-cognitiveservices-speech**：Azure 语音合成引擎

- **elevenlabs**：Elevenlabs 文字转语音转换引擎

- **coqui-TTS**：Coqui的XTTS文本转语音库，用于高质量本地神经TTS

  向[Idiap研究所](https://github.com/idiap)致敬，感谢他们维护了[coqui tts的一个分支](https://github.com/idiap/coqui-ai-TTS)。

- **openai**：与OpenAI的TTS API互动

- **gtts**：谷歌翻译文本转语音转换

