import os

npy_dir = '/data1/ghufran/IMFE/test_features'
frame_dir = '/data1/ghufran/test_frames'

# Get base names (without .npy extension)
npy_files = {os.path.splitext(f)[0] for f in os.listdir(npy_dir) if f.endswith('.npy')}
frame_folders = {f for f in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, f))}

missing = frame_folders - npy_files

print(f"Missing .npy files for {len(missing)} video(s):")
for name in sorted(missing):
    print(name)
