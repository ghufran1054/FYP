
import numpy as np
import os
import onnxruntime as ort
import torch
from collections import OrderedDict
from sklearn.metrics import average_precision_score

def thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False):
    """
    We follow (Shou et al., 2017) and adopt their perframe postprocessing method on THUMOS'14 datset.
    Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
    """

    # Simple temporal smoothing via NMS of 5-frames window
    if smooth:
        prob = np.copy(prediction)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0: -1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0: 2, :], prob[0: -2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        prediction = np.copy(probsmooth)

    # Assign cliff diving (5) as diving (8)
    if switch:
        switch_index = np.where(prediction[:, 5] > prediction[:, 8])[0]
        prediction[switch_index, 8] = prediction[switch_index, 5]

    # Remove ambiguous (21)
    valid_index = np.where(ground_truth[:, 21] != 1)[0]

    return ground_truth[valid_index], prediction[valid_index]


def perframe_average_precision(prediction, ground_truth, class_names,
                               postprocessing=None, metrics='AP'):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)
        
    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        # print('cAP')
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    result['num'] = OrderedDict()
    print(f"NUM FRAMES: {np.sum(ground_truth[:, 1:])}")
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                ap_score = compute_score(ground_truth[:, idx], prediction[:, idx])
                result['per_class_AP'][class_name] = ap_score
                result['num'][class_name] = f'[true: {int(np.sum(ground_truth[:, idx]))}, pred:{int(np.sum(prediction[:,idx]))}, AP:{ap_score*100:.1f}]'
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))

    return result



# Code for actually running the evaluation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = '/data1/ghufran/MiniROAD/data/THUMOS/'
# Find the list of videos in the data path by using any of the folders

video_names = []
for video_name in os.listdir(data_path + 'rgb_kinetics_resnet50/'):
    # Only add if the name contains 'test'
    if 'test' in video_name:
        video_names.append(video_name.split('.')[0])

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
print("Total test videos: ",len(video_names))

pred_scores, gt_targets = [], []

for video_name in video_names:
    rgb_feat = data_path + 'rgb_kinetics_resnet50/' + video_name + '.npy'
    flow_feat = data_path + 'flow_kinetics_bninception/' + video_name + '.npy'
    target_perframe = data_path + 'target_perframe/' + video_name + '.npy'
    # Load all features
    rgb = torch.from_numpy(np.load(rgb_feat)).to(device)
    flow = torch.from_numpy(np.load(flow_feat)).to(device)
    target = torch.from_numpy(np.load(target_perframe)).to(device)

    model_onnx_path = '/data1/ghufran/MiniROAD/mroad.onnx'
    model_onnx_path = '/data1/ghufran/MiniROAD/mroad_flow.onnx'

    session = ort.InferenceSession(model_onnx_path)

    # Input and output names for the ONNX model
    input_rgb_name = session.get_inputs()[0].name
    if 'flow' in model_onnx_path:
        input_flow_name = session.get_inputs()[1].name
        input_h0_name = session.get_inputs()[2].name
    else:
        input_h0_name = session.get_inputs()[1].name
    output_score_name = session.get_outputs()[0].name
    output_h1_name = session.get_outputs()[1].name

    hidden_state = np.zeros((1, 1, 1024), dtype=np.float32)


    rgb = rgb.unsqueeze(0)
    flow = flow.unsqueeze(0)
    predictions = []

    # Setting flow always equal to zero for testing purpose
    flow = torch.zeros_like(flow)

    for i in range(rgb.shape[1]):
        rgb_frame = rgb[:, i:i+1, :]
        flow_frame = flow[:, i:i+1, :]
        if 'flow' in model_onnx_path:
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
        # scores have dimensions [1,1,22] make then [22]
        scores = scores.squeeze()
        predictions.append(scores)

    target = target.cpu().numpy()
    predictions = np.array(predictions)
    pred_scores += list(predictions)
    gt_targets += list(target)
    print("finished video: ", video_name)

print(len(pred_scores))
result = perframe_average_precision(pred_scores, gt_targets, all_class_names, thumos_postprocessing, 'AP')
print(result['mean_AP'])
