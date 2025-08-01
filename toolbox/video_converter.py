import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict
import config
import database_utils
from kaia_cli import KaiaCLI

logger = logging.getLogger(__name__)

def convert_video_to_gif_interactive(cli: KaiaCLI, user_id: str) -> Dict[str, Any]:

    response = ""
    response_type = "video_conversion_error"

    video_files = []
    try:
        for file in os.listdir(config.DOWNLOADS_DIR):
            if file.endswith(".mp4") or file.endswith(".webm"):
                video_files.append(os.path.join(config.DOWNLOADS_DIR, file))
        video_files.sort()

        if not video_files:
            response = f"No .mp4 or .webm files found in {config.DOWNLOADS_DIR}."
            print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
            return {'response': response, 'response_type': response_type}
        else:
            print(f"{config.COLOR_YELLOW}Please select a file to convert:{config.COLOR_RESET}")
            for i, file_path in enumerate(video_files):
                print(f"{i+1}) {os.path.basename(file_path)}")

            try:
                choice = input(f"{config.COLOR_YELLOW}Enter the number of the file: {config.COLOR_RESET}").strip()
                if not choice.isdigit() or not (1 <= int(choice) <= len(video_files)):
                    raise ValueError("Invalid selection.")

                selected_file_path = video_files[int(choice) - 1]
                filename_no_ext = Path(selected_file_path).stem

                input_for_gif = selected_file_path
                output_file = os.path.join(config.DOWNLOADS_DIR, f"{filename_no_ext}.gif")
                temp_mp4_file = None

                if selected_file_path.endswith(".webm"):
                    print(f"{config.COLOR_YELLOW}Detected WebM file. Converting to temporary MP4 first...{config.COLOR_RESET}")
                    temp_mp4_file = os.path.join(config.DOWNLOADS_DIR, f"{filename_no_ext}.temp.mp4")
                    ffmpeg_webm_to_mp4_cmd = [
                        "ffmpeg", "-i", selected_file_path,
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                        "-c:a", "aac", "-b:a", "128k", "-y", temp_mp4_file
                    ]
                    success, stdout, stderr = cli.execute_command(" ".join(ffmpeg_webm_to_mp4_cmd))
                    if not success:
                        response = f"Error converting WebM to MP4. Stderr:\n{stderr}\nStdout:\n{stdout}"
                        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                        if temp_mp4_file and os.path.exists(temp_mp4_file):
                            os.remove(temp_mp4_file)
                        return {'response': response, 'response_type': response_type}
                    input_for_gif = temp_mp4_file

                print(f"{config.COLOR_YELLOW}Converting to GIF...{config.COLOR_RESET}")
                ffmpeg_to_gif_cmd = [
                    "ffmpeg", "-i", input_for_gif, "-y",
                    "-vf", "fps=30,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-loop", "0", output_file
                ]
                success, stdout, stderr = cli.execute_command(" ".join(ffmpeg_to_gif_cmd))

                if temp_mp4_file and os.path.exists(temp_mp4_file):
                    os.remove(temp_mp4_file)
                    print(f"{config.COLOR_BLUE}Cleaned up temporary MP4 file: {temp_mp4_file}{config.COLOR_RESET}")

                if success:
                    response = f"Conversion complete: {output_file}"
                    print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
                    response_type = "video_conversion_success"
                else:
                    response = f"GIF conversion failed. Stderr:\n{stderr}\nStdout:\n{stdout}"
                    print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                    response_type = "video_conversion_error"

            except ValueError as ve:
                response = f"Invalid input: {ve}"
                print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                response_type = "video_conversion_error"
            except Exception as e:
                response = f"An unexpected error occurred during video conversion: {e}"
                print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                response_type = "video_conversion_error"
    except Exception as e:
        response = f"Error listing video files: {e}"
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        response_type = "video_conversion_error"

    return {'response': response, 'response_type': response_type}
