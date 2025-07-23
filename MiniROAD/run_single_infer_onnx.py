import numpy as np
import onnxruntime as ort
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = 'data/THUMOS/'
video_name = 'video_validation_0000202'

rgb_feat = data_path + 'rgb_kinetics_resnet50/' + video_name + '.npy'
# flow_feat = data_path + 'flow_kinetics_bninception/' + video_name + '.npy'
target_perframe = data_path + 'target_perframe/' + video_name + '.npy'
# Load all features
rgb = torch.from_numpy(np.load(rgb_feat)).to(device)
# flow = torch.from_numpy(np.load(flow_feat)).to(device)
target = torch.from_numpy(np.load(target_perframe)).to(device)

session = ort.InferenceSession("mroad.onnx")

hidden_state = np.zeros((1,1,1024), dtype=np.float32)
input_rgb_name = session.get_inputs()[0].name
input_h0_name = session.get_inputs()[1].name
output_score_name = session.get_outputs()[0].name
output_h1_name = session.get_outputs()[1].name


rgb = rgb.unsqueeze(0)
# flow = flow.unsqueeze(0)
predictions = []

for i in range(rgb.shape[1]):
    rgb_frame = rgb[:, i:i+1, :] #this is of size (N, 2048) without torch
    outputs = session.run([output_score_name, output_h1_name], {input_rgb_name: rgb_frame.cpu().numpy(), input_h0_name: hidden_state})
    scores, hidden_state = outputs
    # scores have dimensions [1,1,22] make then [22]
    scores = scores.squeeze()
    predictions.append(scores)

predictions = np.array(predictions)
# In each frame find the class with the highest score
predictions = np.argmax(predictions, axis=1)
print(predictions)