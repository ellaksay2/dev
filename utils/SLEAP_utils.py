import os 

def list_mp4_files(directory):
    # List to store mp4 file names
    mp4_files = []

    # Iterate through files in the given directory
    for filename in os.listdir(directory):
        # Check if file is an mp4
        if filename.endswith('.mp4'):
            mp4_files.append(filename)
    
    return mp4_files


def edit_video_bash(dir, output, mp4_files, brightness=0.0, contrast=1.0): #defaults brightness 0.0 default contrast 1.0
    # Create a bash script file
    script_path = os.path.join(dir, 'edit_videos.sh')
    
    with open(script_path, 'w') as script_file:
        # Write the bash script header
        script_file.write("#!/bin/bash\n\n")

        # Loop through each mp4 file and generate ffmpeg commands
        for video in mp4_files:
            filename = os.path.splitext(video)[0]
            input_path = os.path.join(dir, video)
            output_path = os.path.join(output, f"{filename}_edited.mp4")
            
            # Generate ffmpeg command to adjust brightness and contrast
            ffmpeg_command = f"ffmpeg -i \"{input_path}\" -vf eq=brightness={brightness}:contrast={contrast} \"{output_path}\"\n"

            # Generate ffmpeg command to convert file type
            # ffmpeg_command = f"ffmpeg -y -i \"{input_path}\" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 \"{output_path}\"\n"
            
            # Write the command to the bash script
            script_file.write(ffmpeg_command)

    # Make the bash script executable
    os.chmod(script_path, 0o755)
    print(f"Bash script created: {script_path}")


def edit_video_batch(dir, output, mp4_files, brightness=0.0, contrast=1.0): #defaults brightness 0.0 default contrast 1.0
    # Create a bash script file
    script_path = os.path.join(dir, 'edit_videos.')
    
    with open(script_path, 'w') as script_file:
        # Write the bash script header
        script_file.write("@echo off\n\n")

        # Loop through each mp4 file and generate ffmpeg commands
        for video in mp4_files:
            filename = os.path.splitext(video)[0]
            input_path = os.path.join(dir, video)
            output_path = os.path.join(output, f"{filename}_edited.mp4")
            
            # Generate ffmpeg command to adjust brightness and contrast
            ffmpeg_command = f"ffmpeg -i \"{input_path}\" -vf eq=brightness={brightness}:contrast={contrast} \"{output_path}\"\n"

            # Generate ffmpeg command to convert file type
            # ffmpeg_command = f"ffmpeg -y -i \"{input_path}\" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 \"{output_path}\"\n"
            
            # Write the command to the bash script
            script_file.write(ffmpeg_command)

    # Make the bash script executable
    os.chmod(script_path, 0o755)
    print(f"Batch script created: {script_path}")

def create_inference_bash(directory, mp4_files, model_path, batch_size=4):
    # Calculate the number of scripts needed
    num_scripts = math.ceil(len(mp4_files) / batch_size)

    # Write the bash scripts
    for script_index in range(num_scripts):
        # Create a bash script file 
        script_filename = f"{str(script_index+1).zfill(2)}_inference.sh" #name scripts
        script_path = os.path.join(directory, script_filename)

        with open(script_path, 'w') as script_file:
            # Write the bash script header (LINUX)
            script_file.write("#!/bin/bash\n\n")
            # Write the batch script header (WINDOWS)

            # Write sleap-track commands for a batch of videos
            start_index = script_index * batch_size
            end_index = min(start_index + batch_size, len(mp4_files))

            for i in range(start_index, end_index):
                video_path = os.path.join(directory, mp4_files[i])
                sleap_command = (
                    f"sleap-track \"{video_path}\" "
                    f"-m \"{model_path}\""
                    f"-o \"{video_path}.predictions.slp\n"

                    # Convert .slp files to .h5 files for analysis
                    f"sleap-convert \"{video_path}.predictions.slp\" "
                    f"--format analysis \n"
                )
                # Write the command to the bash script
                script_file.write(sleap_command)

        # Make the bash script executable
        os.chmod(script_path, 0o755)

    print(f"Created {num_scripts} bash scripts in {directory}")


def create_inference_batch(directory, mp4_files, model_path, batch_size=4):
    # Calculate the number of scripts needed
    num_scripts = math.ceil(len(mp4_files) / batch_size)

    # Write the batch scripts
    for script_index in range(num_scripts):
        # Create a batch script file 
        script_filename = f"{str(script_index+1).zfill(2)}_inference.bat" #name scripts

        script_path = os.path.join(directory, script_filename)

        with open(script_path, 'w') as script_file:

            # Write the batch script header (WINDOWS)
            script_file.write("@echo off\n\n")


            # Write sleap-track commands for a batch of videos
            start_index = script_index * batch_size
            end_index = min(start_index + batch_size, len(mp4_files))

            for i in range(start_index, end_index):
                video_path = os.path.join(directory, mp4_files[i])
                sleap_command = (
                    f"sleap-track \"{video_path}\" "
                    f"-m \"{model_path}\""
                    f"-o \"{video_path}.predictions.slp\n"

                    # Convert .slp files to .h5 files for analysis
                    f"sleap-convert \"{video_path}.predictions.slp\" "
                    f"--format analysis \n"
                )
                # Write the command to the bash script
                script_file.write(sleap_command)

        # Make the bash script executable
        os.chmod(script_path, 0o755)

    print(f"Created {num_scripts} bash scripts in {directory}")