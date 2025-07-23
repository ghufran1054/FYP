import numpy as np
import onnxruntime as ort
import torch
import os
import os.path as osp
import matplotlib.pyplot as plt

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path configuration
data_path = '/data1/ghufran/MiniROAD/data/THUMOS/'
video_names = ['video_test_0000004', 'video_validation_0000202']  # Add your video names here
onnx_model_path = "/data1/ghufran/MiniROAD/mroad.onnx"  # Path to the ONNX model
onnx_model_path = "/data1/ghufran/MiniROAD/mroad_flow.onnx"


# Initialize ONNX runtime session
session = ort.InferenceSession(onnx_model_path)

# Input and output names for the ONNX model
input_rgb_name = session.get_inputs()[0].name
if 'flow' in onnx_model_path:
    input_flow_name = session.get_inputs()[1].name
    input_h0_name = session.get_inputs()[2].name
else:
    input_h0_name = session.get_inputs()[1].name
output_score_name = session.get_outputs()[0].name
output_h1_name = session.get_outputs()[1].name

# Initialize hidden state
hidden_state = np.zeros((1, 1, 1024), dtype=np.float32)

# Class names
all_class_names = [
    "Background",
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
    "Ambiguous"
]

# Process each video sequentially
for video_name in video_names:
    ## THis is the folder containing the frames of the video
    frame_folder = '/data1/ghufran/validation_frames' if 'validation' in video_name else '/data1/ghufran/test_frames'
    frame_folder = osp.join(frame_folder, video_name)
    ### Now We have to pick Every 6th frame from the folder and overlay the predictions on the frames and save a video

    print(f"Processing video: {video_name}")

    # Load features for the current video
    rgb_feat = osp.join(data_path, 'rgb_kinetics_resnet50', video_name + '.npy')
    flow_feat = osp.join(data_path, 'flow_kinetics_bninception', video_name + '.npy')
    target_perframe = osp.join(data_path, 'target_perframe', video_name + '.npy')

    rgb = torch.from_numpy(np.load(rgb_feat)).to(device)
    flow = torch.from_numpy(np.load(flow_feat)).to(device)
    target = torch.from_numpy(np.load(target_perframe)).to(device)

    rgb = rgb.unsqueeze(0)
    flow = flow.unsqueeze(0)
    predictions = []

    # Run inference frame by frame
    for i in range(rgb.shape[1]):
        rgb_frame = rgb[:, i:i+1, :]
        flow_frame = flow[:, i:i+1, :]
        if 'flow' in onnx_model_path:
            outputs = session.run(
                [output_score_name, output_h1_name],
                {input_rgb_name: rgb_frame.cpu().numpy(), input_flow_name: flow_frame.cpu().numpy(), input_h0_name: hidden_state}
            )
        else:
            outputs = session.run(
                [output_score_name, output_h1_name],
                {input_rgb_name: rgb_frame.cpu().numpy(), input_h0_name: hidden_state}
            )
        scores, hidden_state = outputs
        scores = scores.squeeze()  # Convert [1, 1, 22] to [22]
        predictions.append(scores)

    prob_val = np.array(predictions)

    # Debug: Print the shape of prob_val
    print(f"Shape of prob_val for {video_name}: {prob_val.shape}")  # Should be (N, 22)

    # Calculate the number of frames where probability > 0.4 for each class
    threshold_prob = 0.5  # Probability threshold
    min_frame_count = 1  # Minimum number of frames required
    class_indices_to_plot = []

    for i, class_name in enumerate(all_class_names):
        # Count the number of frames where probability > threshold_prob
        frame_count_above_threshold = np.sum(prob_val[:, i] > threshold_prob)

        # Check if the count exceeds the minimum frame count
        if frame_count_above_threshold > min_frame_count:
            class_indices_to_plot.append(i)

    # Debug: Print the classes that will be plotted
    print("Classes to plot:", [all_class_names[i] for i in class_indices_to_plot])

    # Plotting the probabilities for selected classes
    plt.figure(figsize=(12, 8))
    for i in class_indices_to_plot:
        plt.plot(prob_val[:, i], label=all_class_names[i])

    plt.xlabel('Frame Number')
    plt.ylabel('Probability')
    plt.title(f'Predicted Probabilities for {video_name} (Classes > {threshold_prob} for > {min_frame_count} frames)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the plot
    output_dir = 'output_plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f'{video_name}_probabilities_filtered_test.png')
    plt.savefig(output_path, bbox_inches='tight')  # Ensure the legend is included in the saved plot
    plt.close()

    print(f"Plot saved to {output_path}")