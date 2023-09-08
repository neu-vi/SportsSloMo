import os

start_clip = 0
end_clip = 8497

for i in range(start_clip, end_clip + 1):
    clip_name = f"YOUR_DATA_PATH/SportsSloMo_video/clip_{i:04d}.mp4"
    output_folder = f"YOUR_DATA_PATH/SportsSloMo_frames/clip_{i:04d}/"
    
    os.makedirs(output_folder, exist_ok=True)
    
    cmd = f"ffmpeg -i {clip_name} -start_number 0 {output_folder}frame_%04d.png"
    os.system(cmd)
    print(f"Processed clip_{i:04d}.mp4")

print("All clips have been decoded.")