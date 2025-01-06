import gradio as gr
import ffmpeg
from transformers import pipeline
import tempfile
import os
import time

# Load transcription pipeline
transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def extract_audio_from_video(video_path):
    """Extract audio from video and save it as a temporary file using FFmpeg."""
    audio_path = video_path.replace(".mp4", "_audio.wav")
    
    # Using FFmpeg to extract audio
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    
    return audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using the loaded transcriber."""
    try:
        transcription = transcriber(audio_path)["text"]
        return transcription
    except Exception as e:
        return f"Error: {str(e)}"

def generate_srt_from_video(video_path, spacing):
    """Generate SRT file from video with adjustable caption spacing."""
    audio_path = extract_audio_from_video(video_path)
    transcription = transcribe_audio(audio_path)
    
    if transcription:
        return generate_srt(transcription, spacing)
    else:
        return "Error in transcription."

def generate_srt(text, spacing):
    """Generate SRT formatted text with adjustable spacing."""
    words = text.split()
    captions = []
    current_caption = []
    current_time = 0
    caption_index = 1
    
    # Adjust timing and caption spacing based on the slider value
    time_per_caption = 3 if spacing < 3 else 5
    max_words_per_caption = 10 if spacing < 3 else 5
    
    for word in words:
        current_caption.append(word)
        
        if len(current_caption) >= max(1, int(spacing)):
            captions.append(f"{caption_index}\n{time.strftime('%H:%M:%S', time.gmtime(current_time))} --> {time.strftime('%H:%M:%S', time.gmtime(current_time + time_per_caption))}\n{' '.join(current_caption)}\n")
            current_caption = []
            current_time += time_per_caption  # Increase the time duration for captions
            caption_index += 1
    
    if current_caption:
        captions.append(f"{caption_index}\n{time.strftime('%H:%M:%S', time.gmtime(current_time))} --> {time.strftime('%H:%M:%S', time.gmtime(current_time + time_per_caption))}\n{' '.join(current_caption)}\n")
    
    # Clean up the extracted audio file
    os.remove(audio_path)

    return "\n".join(captions)

def save_srt_file(srt_content):
    """Save the generated SRT content to a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.srt') as tmpfile:
        tmpfile.write(srt_content.encode('utf-8'))
        tmpfile.close()
        return tmpfile.name  # Return the path to the saved file

def combine_inputs(recorded_video, uploaded_file):
    """Handle combined input options."""
    if recorded_video:
        return recorded_video
    elif uploaded_file:
        return uploaded_file
    else:
        return None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Real-Time Video Captioning")
    gr.Markdown("Upload a video, record, or choose from your gallery, and adjust the slider to control caption spacing.")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Record Video")  # For video recording
            file_input = gr.File(label="Upload Video", file_count="single", type="filepath")  # For file upload with drag-and-drop

        with gr.Column():
            transcribed_text = gr.Textbox(label="Transcribed Text", lines=4)

    caption_slider = gr.Slider(minimum=1, maximum=5, step=1, value=2, label="Caption Spacing")

    # Button to generate captions
    generate_srt_button = gr.Button("Generate Real-Time SRT")
    srt_output = gr.File(label="Download SRT File")

    # Fix for the function call to combine inputs
    generate_srt_button.click(
        generate_srt_from_video, 
        inputs=[combine_inputs(video_input, file_input), caption_slider],  # Call combine_inputs directly here
        outputs=srt_output
    )

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()
