import numpy as np
import onnxruntime as ort
import torch
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path configuration
data_path = '/data1/ghufran/MiniROAD/data/THUMOS/'
video_names = ['video_test_0001325']  # Add your video names here
# video_names = ['video_test_0001201', 'video_test_0001168']
# onnx_model_path = "/data1/ghufran/MiniROAD/mroad.onnx"  # Path to the ONNX model
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
    # This is the folder containing the frames of the video
    frame_folder = '/data1/ghufran/validation_frames' if 'validation' in video_name else '/data1/ghufran/test_frames'
    frame_folder = osp.join(frame_folder, video_name)

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
    threshold_prob = 0.05  # Probability threshold
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

    # Get list of frames in the folder
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])
    frame_files = [osp.join(frame_folder, f) for f in frame_files]

    # Create output video
    output_video_path = osp.join('output_videos_graph', f'{video_name}_overlay_seq_noflow.mp4')
    os.makedirs('output_videos_graph', exist_ok=True)

    # Get frame dimensions
    sample_frame = cv2.imread(frame_files[0])
    height, width, _ = sample_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    graph_height = 120
    dpi = 100
    fig_width = width / dpi
    fig_height = graph_height / dpi
    out = cv2.VideoWriter(output_video_path, fourcc, 4, (width, graph_height + height))  # 10 FPS

    # Create a figure for the graph
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    plt.xlabel('Frame Number')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities')
    plt.grid(True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    

    # Overlay predictions on every 6th frame
    prob_idx = 0
    print("Total frames:", len(frame_files) / 6)
    for i in range(0, len(frame_files), 6):
        frame = cv2.imread(frame_files[i])
        if prob_idx >= prob_val.shape[0]:
            break

        # Clear the previous graph
        plt.clf()

        # Plot the probabilities for selected classes
        for class_idx in class_indices_to_plot:
            class_name = all_class_names[class_idx]
            plt.plot(prob_val[:prob_idx + 1, class_idx], label=class_name)

        # Add legend and labels
        plt.xlabel('Frame Number')
        plt.ylabel('Probability')
        plt.title('Predicted Probabilities')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)



        fig.canvas.draw()


        # Save the graph to a temporary image
        # graph_path = 'temp_graph.png'
        # plt.savefig(graph_path, bbox_inches='tight', dpi=300)

        # Load the graph image
        # graph_img = cv2.imread(graph_path)
        # graph_height, graph_width, _ = graph_img.shape

        # Resize the graph to fit the top section of the frame
        # graph_img_resized = cv2.resize(graph_img, (width, graph_height))

        # Combine the graph and the video frame
        graph_img_resized = np.array(fig.canvas.renderer.buffer_rgba())
        # Convert rgba to rgb
        graph_img_resized = cv2.cvtColor(graph_img_resized[:, :, :3], cv2.COLOR_RGBA2RGB)
        combined_frame = np.vstack((graph_img_resized, frame))

        # Overlay all selected classes and their probabilities
        y_offset = graph_height + 30  # Starting y-coordinate for text
        # for class_idx in class_indices_to_plot:
        #     class_name = all_class_names[class_idx]
        #     prob = prob_val[prob_idx, class_idx]
        #     text = f"{class_name}: {prob:.2f}"
        #     cv2.putText(combined_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #     y_offset += 20  # Move down for the next class

        # Write the combined frame to the output video
        out.write(combined_frame)

        prob_idx += 1

    # Release the video writer
    out.release()

    # Clean up the temporary graph image
    # os.remove(graph_path)

    print(f"Video saved to {output_video_path}")