from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import resize
import os


def cut_and_process_video(input_path, output_dir, start_time="00:00", duration=60, max_outputs=5):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video file
    video = VideoFileClip(input_path)

    # Convert the start time into seconds (MM:SS to seconds)
    minutes, seconds = map(int, start_time.split(":"))
    start_time_in_sec = minutes * 60 + seconds

    # Process video segments
    for i in range(max_outputs):
        # Cut the video into segments
        cut_video = video.subclip(start_time_in_sec + i * duration, start_time_in_sec + (i + 1) * duration)

        # Resize the cut video to fit into 1080x1920 (portrait mode)
        resized_cut_video = cut_video.resize(height=1920)
        resized_cut_video = resized_cut_video.set_width(1080)

        # Blur the original video as the background
        blurred_video = video.subclip(start_time_in_sec + i * duration, start_time_in_sec + (i + 1) * duration).fx(
            resize.resize, 1920).fx(lambda clip: clip.fx(blur, 20))

        # Overlay the cut video on the blurred video
        final_video = blurred_video.set_position("center").overlay(resized_cut_video)

        # Output the final video to the given output directory
        output_path = os.path.join(output_dir, f"output_{i + 1}.mp4")
        final_video.write_videofile(output_path, codec="libx264", fps=24)
        print(f"Video segment {i + 1} saved to {output_path}")


# Example usage
input_video_path = "input_video.mp4"  # Path to the input video
output_directory = "output_videos"  # Directory to save the output
cut_and_process_video(input_video_path, output_directory, start_time="00:30", duration=60, max_outputs=5)
