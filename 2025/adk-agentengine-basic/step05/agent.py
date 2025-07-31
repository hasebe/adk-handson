import io
import logging
import wave

from google import genai
from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext, load_artifacts
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.genai import types
from mcp import StdioServerParameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_GEMINI_2_5_PRO="gemini-2.5-PRO"
MODEL_GEMINI_2_5_FLASH="gemini-2.5-flash"
MODEL_GEMINI_2_5_FLASH_LITE="gemini-2.5-flash-lite"
MODEL_GEMINI_2_5_FLASH_PREVIEW_TTS="gemini-2.5-flash-preview-tts"

client = genai.Client()

async def podcast_speaker(script: str, tool_context: "ToolContext") -> dict[str, str]:
    """
    スクリプトに基づいてポッドキャストの音声を生成します。
    Args:
      script: 音声化するスクリプト。
      tool_context: 現在のツール呼び出しのコンテキスト。
    Returns:
      生成された音声への参照を持つディクショナリ。
    """
    response = client.models.generate_content(
        model=MODEL_GEMINI_2_5_FLASH_PREVIEW_TTS,
        contents=f"""明るくハキハキとしたトーンで2人で会話をしてください。
{script}""",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker='Speaker 1',
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Puck',
                                ),
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker='Speaker 2',
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Zephyr',
                                ),
                            )
                        ),
                    ]
                )
            )
        )
    )

    audio_content = None
    if response and response.candidates:
        candidate = response.candidates[0]
        # candidate.contentが存在するか確認
        if candidate.content and candidate.content.parts:
            part = candidate.content.parts[0]
            # part.inline_dataが存在するか確認
            if part.inline_data:
                audio_content = part.inline_data.data

    if audio_content is None:
        return {
            "status": "failure",
            "detail": "failed to generate podcast audio"
        }

    logger.info(f"Audio length: {len(audio_content)}")

    wav_on_memory = io.BytesIO()
    with wave.open(wav_on_memory, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_content)

    await tool_context.save_artifact(
        filename="podcast.wav",
        artifact=types.Part.from_bytes(data=wav_on_memory.getvalue(), mime_type="audio/wav"),
    )
    return {
        "status": "success",
        "detail": "Audio generated successfully and stored in artifacts.",
        "filename": "podcast.wav",
    }

podcast_writer_agent = LlmAgent(
    name="podcast_writer",
    model=MODEL_GEMINI_2_5_FLASH,
    description="""受け取った内容を MC, 解説者の 2 人で話しているようなポッドキャスト形式の台本に変換します。""",
    instruction="""あなたはプロフェッショナルな放送作家です。受け取った内容を MC, 解説者の 2 人で話しているようなポッドキャスト形式の台本に変換します。

# ルール
- 受け取った内容の重要な点を考えて、必ず台本に含めます。
- 台本は MC を Speaker1, 解説者を Speaker2 として、行頭にどちらが話す内容かがわかるようにします。
- 全体的に楽しい雰囲気で会話が進む台本を作ります。
- 出力には 2 人が話す内容 **のみ** 含めてください。

# 台本の例
Speaker 1: こんにちは。本日は、生成AIの日本の導入状況についてディスカッションできればと思います。
Speaker 2: 生成AIの進化は目を見張るものがありますね。2つのWebサイトから得られた情報をわかり易くお伝えしますね。
Speaker 1: ありがとうございます。まず1つめのWebサイトではどのようなことがわかりましたか？
...
"""
)

podcaster_agent = LlmAgent(
    name="podcaster_assistant",
    model=MODEL_GEMINI_2_5_FLASH,
    description="""受け取った内容を MC, 解説者の 2 人で話しているようなポッドキャスト形式の台本に変換し、それを音声で出力します。""",
    instruction="""あなたはプロフェッショナルなポッドキャストの制作者です。
1. 受け取った内容を `podcast_writer_agent` を利用し、ポッドキャスト形式の台本に変換します。
2. ポッドキャスト形式の台本を `podcast_speaker` ツールを利用して、ポッドキャスト形式の音声に変換して出力します。
""",
    tools=[AgentTool(agent=podcast_writer_agent,), podcast_speaker, load_artifacts],
)

agent = LlmAgent(
    name="learning_assistant",
    model=MODEL_GEMINI_2_5_FLASH,
    description="""ユーザーの学習を助けるために、単体の URL、複数の URL の内容をポッドキャスト形式の音声を生成します。""",
    instruction="""あなたはユーザーの学習を助ける優秀なアシスタントです。単体の URL、複数の URL の内容をポッドキャスト形式の音声を生成します。
1. あなたは URL を与えられたら、`fetch` ツールを利用し、コンテンツを取得します。
   また複数の URL を受け取った場合は `fetch` をそれぞれの URL について呼び出します。
   URL を含まない場合は、ポッドキャスト形式で出力してほしい URL をリクエストしてください。
2. 単一の URL、または複数の URL からコンテンツを取得した場合は、その内容を全て連結します。
3. 連結した内容を `podcaster_agent` に渡し、ポッドキャスト形式の音声を生成します。
""",
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command='uvx',
                    args=[
                        "mcp-server-fetch",
                    ],
                ),
                timeout=10,
            ),
        ),
    ],
    sub_agents=[podcaster_agent,]
)

root_agent = agent