import uuid
import html
import torch
import json

import gradio as gr
import soundfile as sf

from pathlib import Path
from TTS.api import TTS

params = {
    "activate": True,
    "autoplay": True,
    "show_text": False,
    "remove_trailing_dots": False,
    "voice": "Rogger.wav",
    "language": "English",
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
}


SAMPLE_RATE = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_speaker_name = "Rogger"
tts = TTS(model_name=params["model_name"]).to(device)

def gen_voice(string, spk, speed, english):
    string = html.unescape(string)
    short_uuid = str(uuid.uuid4())[:8]
    fl_name="outputs/" + spk + "-" + short_uuid +".wav"
    output_file = Path(fl_name)
    this_dir = str(Path(__file__).parent.resolve())
    tts.tts_to_file(
        text=string,
        speed=speed,
        file_path=output_file,
        speaker_wav=[f"{this_dir}/targets/" +spk + ".wav"],
        language=languages[english]
    )
    return output_file

def update_speakers():
    speakers = {p.stem: str(p) for p in list(Path("targets").glob("*.wav"))}
    return list(speakers.keys())

def update_dropdown(_=None, selected_speaker=default_speaker_name):
    return gr.Dropdown(choices=update_speakers(), value=selected_speaker, label="Select Speaker")

def handle_recorded_audio(audio_data, speaker_dropdown, filename = "user_entered"):
    if not audio_data:
        return speaker_dropdown
    
    sample_rate, audio_content = audio_data
    save_path = f"targets/{filename}.wav"

    sf.write(save_path, audio_content, sample_rate)

    updated_dropdown = update_dropdown(selected_speaker=filename)
    return updated_dropdown

with open(Path("languages.json"), encoding="utf8") as f:
    languages = json.load(f)

with gr.Blocks() as app:
    gr.Markdown("### TTS based Voice Cloning.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=2, label="Speechify this Text",value="Even in the darkest nights, a single spark of hope can ignite the fire of determination within us, guiding us towards a future we dare to dream.")
            speed_slider = gr.Slider(label="Speed", minimum=0.1, maximum=1.99, value=0.8, step=0.01)
            language_dropdown = gr.Dropdown(list(languages.keys()), label="Language/Accent", value="English")

            gr.Markdown("### Speaker Selection and Voice Cloning")
            
            with gr.Row():
                with gr.Column():
                    speaker_dropdown = update_dropdown()
                    refresh_button = gr.Button("Refresh Speakers")
                with gr.Column():
                    filename_input = gr.Textbox(label="Add new Speaker", placeholder="Enter a name for your recording/upload to save as")
                    save_button = gr.Button("Save Below Recording")
                
            refresh_button.click(fn=update_dropdown, inputs=[], outputs=speaker_dropdown)

            with gr.Row():
                record_button = gr.Audio(label="Record Your Voice")
                
            save_button.click(fn=handle_recorded_audio, inputs=[record_button, speaker_dropdown, filename_input], outputs=speaker_dropdown)
            record_button.stop_recording(fn=handle_recorded_audio, inputs=[record_button, filename_input], outputs=speaker_dropdown)
            record_button.upload(fn=handle_recorded_audio, inputs=[record_button, filename_input], outputs=speaker_dropdown)
            
            submit_button = gr.Button("Convert")

        with gr.Column():
            audio_output = gr.Audio()

    submit_button.click(
        fn=gen_voice,
        inputs=[text_input, speaker_dropdown, speed_slider, language_dropdown],
        outputs=audio_output
    )

if __name__ == "__main__":
    app.launch()