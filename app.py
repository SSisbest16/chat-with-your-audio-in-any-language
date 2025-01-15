import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

model = 'whisper-large-v3'
import os
from dotenv import load_dotenv
import streamlit as st


# Display debug information in Streamlit
if GROQ_API_KEY:
    st.write(f"GROQ_API_KEY loaded successfully: {GROQ_API_KEY}")
else:
    st.error("GROQ_API_KEY not found! Please check your .env file.")

# Audio Transcription Function
def audio_to_text(filepath):
    with open(filepath, 'rb') as file:
        translation = client.audio.translations.create(
            file=(filepath, file.read()),
            model='whisper-large-v3'
        )
    return translation.text

# Chat Completion Function
def transcript_chat_completion(client, transcript, user_question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f'''Use this transcript or transcripts to answer any user questions, citing specific quotes:

                {transcript}
                '''
            },
            {
                "role": "user",
                "content": user_question,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Streamlit Interface
st.title("Audio Q&A with Groq")

# File Uploader
audio_file = st.file_uploader("Upload an audio file (MP3, WAV)", type=["mp3", "wav"])
if audio_file is not None:
    # Save the uploaded file temporarily
    audio_path = "uploaded_audio.mp3"
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
    
    # Transcribe the audio
    st.write("Transcribing audio...")
    try:
        transcription = audio_to_text(audio_path)
        st.success("Transcription completed!")
        st.text_area("Transcript", transcription, height=200)

        # Q&A Section
        st.write("### Ask questions about the audio content")
        user_question = st.text_input("Your Question")
        if user_question:
            st.write("Answering your question...")
            try:
                answer = transcript_chat_completion(client, transcription, user_question)
                st.success(f"Answer: {answer}")
            except Exception as e:
                st.error(f"Error while answering: {e}")
    except Exception as e:
        st.error(f"Error during transcription: {e}")

