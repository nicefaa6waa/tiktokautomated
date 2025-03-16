from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
import os
import cv2
import numpy as np
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import json
from googletrans import Translator
import wave
import time
import shutil
import subprocess


def blur_image(image):
    return cv2.GaussianBlur(image, (21, 21), 20)


def time_to_seconds(time_str):
    """Convert MM:SS or H:MM:SS format to seconds."""
    parts = list(map(int, time_str.split(":")))
    if len(parts) == 2:  # MM:SS
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:  # H:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError("Time format must be MM:SS or H:MM:SS")


def wrap_text(text, max_width, fontsize):
    """Wrap text if it exceeds max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    for word in words:
        word_width = len(word) * fontsize * 0.6
        if current_width + word_width > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width
        else:
            current_line.append(word)
            current_width += word_width + (fontsize * 0.3)
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def generate_subtitles_en(audio_path, duration):
    """Generate English subtitles using speech recognition."""
    recognizer = sr.Recognizer()
    print(f"Processing audio file: {audio_path}")
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    print(f"Expected audio duration: {duration} seconds")
    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed: {text}")
        return [(0, duration, text)]
    except sr.UnknownValueError:
        print("Could not understand audio")
        return []
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return []


def generate_subtitles_jp(audio_path, duration):
    """Generate Japanese subtitles using Vosk and translate to English."""
    if not os.path.exists("vosk-model-small-ja-0.22"):
        print(
            "Please download the Vosk Japanese model 'vosk-model-small-ja-0.22' and place it in the script directory.")
        return []
    model = Model("vosk-model-small-ja-0.22")
    recognizer = KaldiRecognizer(model, 16000)
    subtitles = []
    with wave.open(audio_path, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if "result" in result:
                    for word in result["result"]:
                        start = word["start"]
                        end = word["end"]
                        text = word["word"]
                        subtitles.append((start, end, text))
    if subtitles:
        combined_text = " ".join([s[2] for s in subtitles])
        start_time = subtitles[0][0]
        end_time = subtitles[-1][1]
        translator = Translator()
        translated_text = translator.translate(combined_text, src="ja", dest="en").text
        return [(start_time, end_time, translated_text)]
    return []


def add_subtitles_to_video(video_clip, subtitles, video_width=1080):
    """Add subtitles to the video clip."""
    subtitle_clips = []
    for start, end, text in subtitles:
        wrapped_text = wrap_text(text, 900, 30)
        subtitle = TextClip(
            wrapped_text,
            fontsize=30,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(video_width - 100, None),
            align="center"
        )
        subtitle = subtitle.set_position(("center", 480)).set_start(start).set_end(end)
        subtitle_clips.append(subtitle)
    return CompositeVideoClip([video_clip] + subtitle_clips)


def move_and_upload_videos(source_dir, upload_dir, username="animefandaily", caption="Uploading Daily #SoloLeveling #SungJinwoo #fyp #anime #Jinwoo #Beru"):
    """Move videos to upload_dir, upload them with delays, and delete after uploading."""
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    video_files = [f for f in os.listdir(source_dir) if f.endswith(".mp4")]
    video_files.sort()  # Ensure order: output_1.mp4, output_2.mp4, output_3.mp4

    for video in video_files:
        source_path = os.path.join(source_dir, video)
        dest_path = os.path.join(upload_dir, video)
        shutil.move(source_path, dest_path)
        print(f"Moved {video} to {upload_dir}")

        cli_dir = "uploader"  # Directory containing cli.py
        upload_command = [
            "python", "cli.py", "upload",
            "-u", username,
            "-v", video,
            "-t", caption
        ]
        print(f"Running: {' '.join(upload_command)} from {cli_dir}")
        subprocess.run(upload_command, cwd=cli_dir, check=True)
        print(f"Uploaded {video}")

        os.remove(dest_path)
        print(f"Deleted {dest_path}")

        if video != video_files[-1]:
            print("Waiting 10 minutes before next upload...")
            time.sleep(600)


def cut_and_process_video(input_path, output_dir, language, total_output_count, add_subtitles_flag=False,
                          start_time="00:00", duration=60, max_outputs=None, intro_start=None, intro_end=None,
                          outro_start=None, outro_end=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = VideoFileClip(input_path)
    total_duration = video.duration

    start_time_in_sec = time_to_seconds(start_time)
    intro_start_in_sec = 0 if intro_start == "start" else (time_to_seconds(intro_start) if intro_start else None)
    intro_end_in_sec = time_to_seconds(intro_end) if intro_end else None
    outro_start_in_sec = time_to_seconds(outro_start) if outro_start else None
    outro_end_in_sec = total_duration if outro_end == "end" else (time_to_seconds(outro_end) if outro_end else None)

    current_time = start_time_in_sec
    output_count = 0
    recent_outputs = []

    try:
        while current_time < total_duration and (max_outputs is None or output_count < max_outputs):
            segment_end = current_time + duration

            if intro_start_in_sec is not None and intro_end_in_sec:
                if current_time < intro_end_in_sec:
                    if segment_end <= intro_end_in_sec:
                        current_time = segment_end
                        continue
                    else:
                        segment_end = intro_end_in_sec
                        current_time = intro_end_in_sec
                        continue

            if outro_start_in_sec:
                if current_time < outro_start_in_sec < segment_end:
                    segment_end = outro_start_in_sec
                elif current_time >= outro_start_in_sec:
                    break

            if segment_end > total_duration:
                segment_end = total_duration

            cut_video = video.subclip(current_time, segment_end)
            temp_output_path_0 = os.path.join(output_dir, f"temp_output_{output_count + 1}.0.mp4")
            cut_video.write_videofile(
                temp_output_path_0,
                codec="libx264",
                fps=24,
                temp_audiofile=os.path.join(output_dir, f"temp_audio_{output_count + 1}_0.mp3")
            )
            temp_audio_0 = os.path.join(output_dir, f"temp_audio_{output_count + 1}_0.mp3")
            if os.path.exists(temp_audio_0):
                os.remove(temp_audio_0)

            temp_video_0 = VideoFileClip(temp_output_path_0)
            resized_temp_video_1 = temp_video_0.resize((1080, 1920))
            blurred_temp_video_1 = resized_temp_video_1.fl_image(blur_image)
            temp_output_path_1 = os.path.join(output_dir, f"temp_output_{output_count + 1}.1.mp4")
            blurred_temp_video_1.write_videofile(
                temp_output_path_1,
                codec="libx264",
                fps=24,
                temp_audiofile=os.path.join(output_dir, f"temp_audio_{output_count + 1}_1.mp3")
            )
            temp_audio_1 = os.path.join(output_dir, f"temp_audio_{output_count + 1}_1.mp3")
            if os.path.exists(temp_audio_1):
                os.remove(temp_audio_1)

            blurred_temp_video_1 = VideoFileClip(temp_output_path_1)
            width = 1080
            height = int(temp_video_0.h * (width / temp_video_0.w))
            resized_temp_video_0 = temp_video_0.resize((width, height))

            final_video = CompositeVideoClip(
                [blurred_temp_video_1, resized_temp_video_0.set_position(("center", "center"))])

            if add_subtitles_flag:
                audio_path = os.path.join(output_dir, f"temp_audio_{output_count + 1}.wav")
                print(f"Extracting audio to: {audio_path}")
                cut_video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
                print(f"Audio extracted successfully: {os.path.exists(audio_path)}")
                if language == "en":
                    subtitles = generate_subtitles_en(audio_path, segment_end - current_time)
                elif language == "jp":
                    subtitles = generate_subtitles_jp(audio_path, segment_end - current_time)
                if subtitles:
                    final_video = add_subtitles_to_video(final_video, subtitles)
                if os.path.exists(audio_path):
                    os.remove(audio_path)

            output_path = os.path.join(output_dir, f"output_{output_count + 1}.mp4")
            final_video.write_videofile(
                output_path,
                codec="libx264",
                fps=24,
                temp_audiofile=os.path.join(output_dir, f"temp_audio_{output_count + 1}_final.mp3")
            )
            temp_audio_final = os.path.join(output_dir, f"temp_audio_{output_count + 1}_final.mp3")
            if os.path.exists(temp_audio_final):
                os.remove(temp_audio_final)
            print(f"Video segment {output_count + 1} saved to {output_path}")

            os.remove(temp_output_path_0)
            os.remove(temp_output_path_1)

            current_time = segment_end
            output_count += 1
            total_output_count[0] += 1
            recent_outputs.append(output_path)

            # Upload after every 3 outputs
            if total_output_count[0] % 3 == 0 and total_output_count[0] > 0:
                print(f"\nGenerated {total_output_count[0]} total output segments. Uploading...")
                move_and_upload_videos(output_dir, os.path.join("uploader", "VideosDirPath"))
                print("Resuming processing...")

    finally:
        video.close()
        if 'temp_video_0' in locals():
            temp_video_0.close()
        if 'blurred_temp_video_1' in locals():
            blurred_temp_video_1.close()
        if 'final_video' in locals():
            final_video.close()

    return output_count


def scan_and_select_videos(input_dir):
    """Scan input directory for .mkv files and let user exclude some, sorted alphabetically."""
    video_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mkv')])
    if not video_files:
        raise ValueError(f"No .mkv files found in {input_dir}")
    print("\nAvailable videos:")
    for i, video in enumerate(video_files, 1):
        print(f"{i}) {video}")
    exclude_input = input("Enter numbers of videos to exclude (e.g., 1,2,3), or leave blank to include all: ").strip()
    if exclude_input:
        exclude_indices = [int(i) - 1 for i in exclude_input.split(',') if i.strip().isdigit()]
        selected_videos = [os.path.join(input_dir, video_files[i]) for i in range(len(video_files)) if
                           i not in exclude_indices]
    else:
        selected_videos = [os.path.join(input_dir, video) for video in video_files]
    if not selected_videos:
        raise ValueError("No videos selected after exclusion.")
    return selected_videos


def collect_intro_outro_times(video_list):
    """Collect intro and outro times for all videos beforehand."""
    timing_data = {}
    for video_path in video_list:
        print(f"\nSetting times for {video_path}")
        intro_start = input(
            f"Enter intro start time for {video_path} (e.g., 'start', 01:00, leave blank if no intro): ").strip() or None
        intro_end = None
        if intro_start:
            if intro_start == "start":
                intro_end = input(f"Enter intro end time for {video_path} (e.g., 01:00): ").strip()
            else:
                intro_end = input(f"Enter intro end time for {video_path} (e.g., 02:15): ").strip()
            if not intro_end:
                raise ValueError("Intro end time must be provided if intro start time is specified.")
        outro_start = input(
            f"Enter outro start time for {video_path} (e.g., 22:00, leave blank if no outro): ").strip() or None
        outro_end = None
        if outro_start:
            outro_end = input(
                f"Enter outro end time for {video_path} (e.g., 23:00, 'end' for video end): ").strip() or "end"
        timing_data[video_path] = {
            "intro_start": intro_start,
            "intro_end": intro_end,
            "outro_start": outro_start,
            "outro_end": outro_end
        }
    return timing_data


def process_multiple_videos(video_list, base_output_dir, project_name, start_time="00:00", duration=60):
    project_dir = os.path.join(base_output_dir, project_name)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    timing_data = collect_intro_outro_times(video_list)
    total_output_count = [0]  # Track total outputs across all videos

    for video_number, video_path in enumerate(video_list, start=1):
        video_output_dir = os.path.join(project_dir, str(video_number))
        print(f"\nProcessing {video_path}")
        max_outputs_input = input(
            f"How many outputs do you want per video for {video_path}? (e.g., 5 or leave blank (until video ends)): ").strip()
        max_outputs = int(max_outputs_input) if max_outputs_input else None
        language = input(
            f"What language is {video_path} in? (jp for Japanese, en for English, other for other languages): ").strip().lower()
        if language not in ["jp", "en", "other"]:
            raise ValueError("Language must be 'jp', 'en', or 'other'.")
        add_subtitles_flag = False
        if language == "en":
            add_subtitles = input(f"Do you want to add subtitles for {video_path}? (yes/no): ").strip().lower()
            add_subtitles_flag = add_subtitles == "yes"
        elif language == "jp":
            print(f"Video is Japanese, subtitles will be added in English.")
            add_subtitles_flag = True
        elif language == "other":
            print("Only English and Japanese subtitles are supported. Subtitles will not be added.")

        times = timing_data[video_path]
        cut_and_process_video(video_path, video_output_dir, language, total_output_count, add_subtitles_flag,
                              start_time, duration, max_outputs, times["intro_start"], times["intro_end"],
                              times["outro_start"], times["outro_end"])


if __name__ == "__main__":
    project_name = input("Enter project name (e.g., Solo Leveling): ").strip()
    if not project_name:
        raise ValueError("Project name cannot be empty.")
    input_directory = "input_videos"
    input_videos = scan_and_select_videos(input_directory)
    base_output_directory = "video_output"  # Changed from output_videos to video_output
    process_multiple_videos(input_videos, base_output_directory, project_name, start_time="00:30", duration=60)