import os
import tempfile
import gradio as gr
from groq import Groq
from gtts import gTTS
import whisper

# ── Load API key from HF Secrets (injected as env variable) ──────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY secret not set! Add it in Space Settings → Secrets.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Load Whisper model once at startup ───────────────────────────────────────
# Use "tiny" for faster cold start on HF free CPU tier
# Switch to "base" or "small" if you need better accuracy
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("✅ Whisper ready!")


# ── Core pipeline functions ───────────────────────────────────────────────────

def speech_to_text(audio_path):
    """Transcribe audio file to text using Whisper."""
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def get_llm_response(user_text, chat_history):
    """Get response from LLaMA 3.3 70B via Groq."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, friendly voice assistant. "
                "Keep responses concise and conversational — they will be spoken aloud. "
                "Avoid bullet points, markdown formatting, or special characters."
            ),
        }
    ]

    for human_msg, ai_msg in chat_history:
        messages.append({"role": "user", "content": human_msg})
        messages.append({"role": "assistant", "content": ai_msg})

    messages.append({"role": "user", "content": user_text})

    response = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content


def text_to_speech(text):
    """Convert text to speech using gTTS. Returns path to .mp3 file."""
    tts = gTTS(text=text, lang="en", slow=False)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        return f.name


def voice_pipeline(audio, chat_history):
    """Full pipeline: audio → Whisper → Groq LLaMA → gTTS → audio."""
    if audio is None:
        return chat_history, None, "⚠️ No audio received. Please record something."

    # Step 1: STT
    user_text = speech_to_text(audio)
    if not user_text:
        return chat_history, None, "⚠️ Could not understand. Please speak clearly and try again."

    # Step 2: LLM
    ai_response = get_llm_response(user_text, chat_history)

    # Step 3: TTS
    audio_response_path = text_to_speech(ai_response)

    # Update history
    chat_history.append((user_text, ai_response))

    transcript = f"🎙️ You: {user_text}\n\n🤖 AI: {ai_response}"
    return chat_history, audio_response_path, transcript


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="🎙️ Voice AI Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🎙️ Voice AI Assistant")
    gr.Markdown(
        "Powered by **Whisper** (Speech-to-Text) · "
        "**LLaMA 3.3 70B via Groq** (LLM) · "
        "**gTTS** (Text-to-Speech) — 100% Free"
    )

    chat_history_state = gr.State([])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎤 Speak to the AI")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record your message",
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        with gr.Column():
            gr.Markdown("### 🔊 AI Response")
            audio_output = gr.Audio(
                label="AI Voice",
                autoplay=True,
            )

    transcript_box = gr.Textbox(
        label="📝 Last Exchange",
        lines=4,
        interactive=False,
    )

    chatbot = gr.Chatbot(label="💬 Conversation History", height=300)

    submit_btn.click(
        fn=voice_pipeline,
        inputs=[audio_input, chat_history_state],
        outputs=[chat_history_state, audio_output, transcript_box],
    ).then(
        fn=lambda h: h,
        inputs=[chat_history_state],
        outputs=[chatbot],
    )

    clear_btn.click(
        fn=lambda: ([], None, "", []),
        outputs=[chat_history_state, audio_output, transcript_box, chatbot],
    )

# HF Spaces requires server_name="0.0.0.0" — do NOT use share=True here
demo.launch(server_name="0.0.0.0", server_port=7860)
