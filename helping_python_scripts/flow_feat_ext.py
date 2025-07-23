import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
video_list = {
"train_session_set": 
    ["video_validation_0000690", "video_validation_0000288", "video_validation_0000289", "video_validation_0000416", "video_validation_0000282", "video_validation_0000283", "video_validation_0000281", "video_validation_0000286", "video_validation_0000287", "video_validation_0000284", "video_validation_0000285", "video_validation_0000202", "video_validation_0000203", "video_validation_0000201", "video_validation_0000206", "video_validation_0000207", "video_validation_0000204", "video_validation_0000205", "video_validation_0000790", "video_validation_0000208", "video_validation_0000209", "video_validation_0000420", "video_validation_0000364", "video_validation_0000853", "video_validation_0000950", "video_validation_0000937", "video_validation_0000367", "video_validation_0000290", "video_validation_0000210", "video_validation_0000059", "video_validation_0000058", "video_validation_0000057", "video_validation_0000056", "video_validation_0000055", "video_validation_0000054", "video_validation_0000053", "video_validation_0000052", "video_validation_0000051", "video_validation_0000933", "video_validation_0000949", "video_validation_0000948", "video_validation_0000945", "video_validation_0000944", "video_validation_0000947", "video_validation_0000946", "video_validation_0000941", "video_validation_0000940", "video_validation_0000190", "video_validation_0000942", "video_validation_0000261", "video_validation_0000262", "video_validation_0000263", "video_validation_0000264", "video_validation_0000265", "video_validation_0000266", "video_validation_0000267", "video_validation_0000268", "video_validation_0000269", "video_validation_0000989", "video_validation_0000060", "video_validation_0000370", "video_validation_0000938", "video_validation_0000935", "video_validation_0000668", "video_validation_0000669", "video_validation_0000664", "video_validation_0000665", "video_validation_0000932", "video_validation_0000667", "video_validation_0000934", "video_validation_0000661", "video_validation_0000662", "video_validation_0000663", "video_validation_0000181", "video_validation_0000180", "video_validation_0000183", "video_validation_0000182", "video_validation_0000185", "video_validation_0000184", "video_validation_0000187", "video_validation_0000186", "video_validation_0000189", "video_validation_0000188", "video_validation_0000936", "video_validation_0000270", "video_validation_0000854", "video_validation_0000178", "video_validation_0000179", "video_validation_0000174", "video_validation_0000175", "video_validation_0000176", "video_validation_0000177", "video_validation_0000170", "video_validation_0000171", "video_validation_0000172", "video_validation_0000173", "video_validation_0000670", "video_validation_0000419", "video_validation_0000943", "video_validation_0000485", "video_validation_0000369", "video_validation_0000368", "video_validation_0000318", "video_validation_0000319", "video_validation_0000415", "video_validation_0000414", "video_validation_0000413", "video_validation_0000412", "video_validation_0000411", "video_validation_0000311", "video_validation_0000312", "video_validation_0000313", "video_validation_0000314", "video_validation_0000315", "video_validation_0000316", "video_validation_0000317", "video_validation_0000418", "video_validation_0000365", "video_validation_0000482", "video_validation_0000169", "video_validation_0000168", "video_validation_0000167", "video_validation_0000166", "video_validation_0000165", "video_validation_0000164", "video_validation_0000163", "video_validation_0000162", "video_validation_0000161", "video_validation_0000160", "video_validation_0000857", "video_validation_0000856", "video_validation_0000855", "video_validation_0000366", "video_validation_0000488", "video_validation_0000489", "video_validation_0000851", "video_validation_0000484", "video_validation_0000361", "video_validation_0000486", "video_validation_0000487", "video_validation_0000481", "video_validation_0000910", "video_validation_0000483", "video_validation_0000363", "video_validation_0000990", "video_validation_0000939", "video_validation_0000362", "video_validation_0000987", "video_validation_0000859", "video_validation_0000787", "video_validation_0000786", "video_validation_0000785", "video_validation_0000784", "video_validation_0000783", "video_validation_0000782", "video_validation_0000781", "video_validation_0000981", "video_validation_0000983", "video_validation_0000982", "video_validation_0000985", "video_validation_0000984", "video_validation_0000417", "video_validation_0000788", "video_validation_0000152", "video_validation_0000153", "video_validation_0000151", "video_validation_0000156", "video_validation_0000157", "video_validation_0000154", "video_validation_0000155", "video_validation_0000158", "video_validation_0000159", "video_validation_0000901", "video_validation_0000903", "video_validation_0000902", "video_validation_0000905", "video_validation_0000904", "video_validation_0000907", "video_validation_0000906", "video_validation_0000909", "video_validation_0000908", "video_validation_0000490", "video_validation_0000860", "video_validation_0000858", "video_validation_0000988", "video_validation_0000320", "video_validation_0000688", "video_validation_0000689", "video_validation_0000686", "video_validation_0000687", "video_validation_0000684", "video_validation_0000685", "video_validation_0000682", "video_validation_0000683", "video_validation_0000681", "video_validation_0000789", "video_validation_0000986", "video_validation_0000931", "video_validation_0000852", "video_validation_0000666"],
    "test_session_set": 
    ["video_test_0000292", "video_test_0001078", "video_test_0000896", "video_test_0000897", "video_test_0000950", "video_test_0001159", "video_test_0001079", "video_test_0000807", "video_test_0000179", "video_test_0000173", "video_test_0001072", "video_test_0001075", "video_test_0000767", "video_test_0001076", "video_test_0000007", "video_test_0000006", "video_test_0000556", "video_test_0001307", "video_test_0001153", "video_test_0000718", "video_test_0000716", "video_test_0001309", "video_test_0000714", "video_test_0000558", "video_test_0001267", "video_test_0000367", "video_test_0001324", "video_test_0000085", "video_test_0000887", "video_test_0001281", "video_test_0000882", "video_test_0000671", "video_test_0000964", "video_test_0001164", "video_test_0001114", "video_test_0000771", "video_test_0001163", "video_test_0001118", "video_test_0001201", "video_test_0001040", "video_test_0001207", "video_test_0000723", "video_test_0000569", "video_test_0000672", "video_test_0000673", "video_test_0000278", "video_test_0001162", "video_test_0000405", "video_test_0000073", "video_test_0000560", "video_test_0001276", "video_test_0000270", "video_test_0000273", "video_test_0000374", "video_test_0000372", "video_test_0001168", "video_test_0000379", "video_test_0001446", "video_test_0001447", "video_test_0001098", "video_test_0000873", "video_test_0000039", "video_test_0000442", "video_test_0001219", "video_test_0000762", "video_test_0000611", "video_test_0000617", "video_test_0000615", "video_test_0001270", "video_test_0000740", "video_test_0000293", "video_test_0000504", "video_test_0000505", "video_test_0000665", "video_test_0000664", "video_test_0000577", "video_test_0000814", "video_test_0001369", "video_test_0001194", "video_test_0001195", "video_test_0001512", "video_test_0001235", "video_test_0001459", "video_test_0000691", "video_test_0000765", "video_test_0001452", "video_test_0000188", "video_test_0000591", "video_test_0001268", "video_test_0000593", "video_test_0000864", "video_test_0000601", "video_test_0001135", "video_test_0000004", "video_test_0000903", "video_test_0000285", "video_test_0001174", "video_test_0000046", "video_test_0000045", "video_test_0001223", "video_test_0001358", "video_test_0001134", "video_test_0000698", "video_test_0000461", "video_test_0001182", "video_test_0000450", "video_test_0000602", "video_test_0001229", "video_test_0000989", "video_test_0000357", "video_test_0001039", "video_test_0000355", "video_test_0000353", "video_test_0001508", "video_test_0000981", "video_test_0000242", "video_test_0000854", "video_test_0001484", "video_test_0000635", "video_test_0001129", "video_test_0001339", "video_test_0001483", "video_test_0001123", "video_test_0001127", "video_test_0000689", "video_test_0000756", "video_test_0001431", "video_test_0000129", "video_test_0001433", "video_test_0001343", "video_test_0000324", "video_test_0001064", "video_test_0001531", "video_test_0001532", "video_test_0000413", "video_test_0000991", "video_test_0001255", "video_test_0000464", "video_test_0001202", "video_test_0001080", "video_test_0001081", "video_test_0000847", "video_test_0000028", "video_test_0000844", "video_test_0000622", "video_test_0000026", "video_test_0001325", "video_test_0001496", "video_test_0001495", "video_test_0000624", "video_test_0000724", "video_test_0001409", "video_test_0000131", "video_test_0000448", "video_test_0000444", "video_test_0000443", "video_test_0001038", "video_test_0000238", "video_test_0001527", "video_test_0001522", "video_test_0000051", "video_test_0001058", "video_test_0001391", "video_test_0000429", "video_test_0000426", "video_test_0000785", "video_test_0000786", "video_test_0001314", "video_test_0000392", "video_test_0000423", "video_test_0001146", "video_test_0001313", "video_test_0001008", "video_test_0001247", "video_test_0000737", "video_test_0001319", "video_test_0000308", "video_test_0000730", "video_test_0000058", "video_test_0000538", "video_test_0001556", "video_test_0000113", "video_test_0000626", "video_test_0000839", "video_test_0000220", "video_test_0001389", "video_test_0000437", "video_test_0000940", "video_test_0000211", "video_test_0000946", "video_test_0001558", "video_test_0000796", "video_test_0000062", "video_test_0000793", "video_test_0000987", "video_test_0001066", "video_test_0000412", "video_test_0000798", "video_test_0001549", "video_test_0000011", "video_test_0001257", "video_test_0000541", "video_test_0000701", "video_test_0000250", "video_test_0000254", "video_test_0000549", "video_test_0001209", "video_test_0001463", "video_test_0001460", "video_test_0000319", "video_test_0001468", "video_test_0000846", "video_test_0001292"]
}
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization for grayscale
mean_std = 128.0 / 255.0
transform = transforms.Compose([
    # Also take a center crop to 224x224
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Converts to (1, H, W) for grayscale
    transforms.Normalize(mean=[mean_std], std=[mean_std])
])

def process_folder(folder_path, model, out_folder, batch_size=1000):
    frame_files = sorted([f for f in os.listdir(folder_path) if 'x' in f])
    num_frames = len(frame_files)

    if num_frames > 3000:
        print(f"Skipping {folder_path} with {num_frames} frames (more than 5000)")
        return
    if num_frames == 0:
        print(f"No frames found in {folder_path}")
        return
    features = []
    batch_inputs = []
    count = 0
    for idx in range(0, num_frames, 6):
        imgs = []
        valid = True
        start_idx = 0
        count += 1
        for i in range(start_idx, start_idx + 5):
            frame_idx = idx + i
            x_path = os.path.join(folder_path, f"x_{frame_idx:05d}.jpg")
            y_path = os.path.join(folder_path, f"y_{frame_idx:05d}.jpg")
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                # print(f"Missing frames in {folder_path} for index {idx}")
                valid = False
                break

            # Load images using openCV
            x_img = Image.open(x_path).convert('L')  # Convert to grayscale
            y_img = Image.open(y_path).convert('L')  # Convert to grayscale
            x_img = transform(x_img)  # (1, H, W)
            y_img = transform(y_img)  # (1, H, W)
            imgs.append(x_img)
            imgs.append(y_img)
        # imgs.extend(imgs_x)
        # imgs.extend(imgs_y)
            
            # Maintaining alternating way x and y

        if not valid:
            # Skip this batch if any frame is missing
            continue

        input_tensor = torch.cat(imgs, dim=0).unsqueeze(0)  # (1, 10, H, W)
        batch_inputs.append(input_tensor)

        if len(batch_inputs) == batch_size:
            input_batch = torch.cat(batch_inputs, dim=0).to(device)  # (B, 10, H, W)

            # Add one more dime because our model needs 1, B, 10, H, W
            input_batch = input_batch.unsqueeze(0)  # (B, 1, 10, H, W)
            # exit()
            with torch.no_grad():
                feats = model(input_batch)  # (B, 2048, 7, 7)
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1).cpu()  # (B, 2048)
                features.append(feats)
            batch_inputs = []

    # Process leftover batch
    if batch_inputs:
        input_batch = torch.cat(batch_inputs, dim=0).to(device)
        input_batch = input_batch.unsqueeze(0)
        with torch.no_grad():
            feats = model(input_batch)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1).cpu()
            features.append(feats)

    if features:
        features_tensor = torch.cat(features, dim=0)  # (N, 2048)
        # Create the output directory if it doesn't exist
        os.makedirs(out_folder, exist_ok=True)
        output_path = os.path.basename(folder_path.rstrip("/")) + ".npy"
        np.save(out_folder + output_path, features_tensor.numpy())


        # Open a file with path /data1/ghufran/THUMOS/target_perframe/output_path"
        target = np.load('/data1/ghufran/THUMOS/target_perframe/' + output_path)

        # Check if the length of features_tensor and target is same
        if features_tensor.shape[0] != target.shape[0]:
            # Open a file called report.txt mode 'a' and write the output_path and the difference in length
            with open('/data1/ghufran/helping_python_scripts/report.txt', 'a') as f:
                f.write(f"Length mismatch for {output_path}: {features_tensor.shape[0]} vs {target.shape[0]}\n")

        # print(f"Saved: {out_folder + output_path}")
        # print("total count: ", count)







# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine import dump, list_from_file, load
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import torch



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 feature extraction')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_prefix', type=str, help='output prefix')
    parser.add_argument(
        '--video-list', type=str, default=None, help='video file list')
    parser.add_argument(
        '--video-root', type=str, default=None, help='video root directory')
    parser.add_argument(
        '--spatial-type',
        type=str,
        default='avg',
        choices=['avg', 'max', 'keep'],
        help='Pooling type in spatial dimension')
    parser.add_argument(
        '--temporal-type',
        type=str,
        default='keep',
        choices=['avg', 'max', 'keep'],
        help='Pooling type in temporal dimension')
    parser.add_argument(
        '--long-video-mode',
        action='store_true',
        help='Perform long video inference to get a feature list from a video')
    parser.add_argument(
        '--clip-interval',
        type=int,
        default=None,
        help='Clip interval for Clip interval of adjacent center of sampled '
        'clips, used for long video inference')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=None,
        help='Temporal interval of adjacent sampled frames, used for long '
        'video long video inference')
    parser.add_argument(
        '--multi-view',
        action='store_true',
        help='Perform multi view inference')
    parser.add_argument(
        '--dump-score',
        action='store_true',
        help='Dump predict scores rather than features')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    # -------------------- Feature Head --------------------
    if not args.dump_score:
        backbone_type2name = dict(
            ResNet3dSlowFast='slowfast',
            MobileNetV2TSM='tsm',
            ResNetTSM='tsm',
        )

        if cfg.model.type == 'RecognizerGCN':
            backbone_name = 'gcn'
        else:
            backbone_name = backbone_type2name.get(cfg.model.backbone.type)
        num_segments = None
        if backbone_name == 'tsm':
            for idx, transform in enumerate(test_pipeline):
                if transform.type == 'UntrimmedSampleFrames':
                    clip_len = transform['clip_len']
                    continue
                elif transform.type == 'SampleFrames':
                    clip_len = transform['num_clips']
            num_segments = cfg.model.backbone.get('num_segments', 8)
            assert num_segments == clip_len, \
                f'num_segments and clip length must same for TSM, but got ' \
                f'num_segments {num_segments} clip_len {clip_len}'
            if cfg.model.test_cfg is not None:
                max_testing_views = cfg.model.test_cfg.get(
                    'max_testing_views', num_segments)
                assert max_testing_views % num_segments == 0, \
                    'tsm needs to infer with batchsize of multiple ' \
                    'of num_segments.'

        spatial_type = None if args.spatial_type == 'keep' else \
            args.spatial_type
        temporal_type = None if args.temporal_type == 'keep' else \
            args.temporal_type
        feature_head = dict(
            type='FeatureHead',
            spatial_type=spatial_type,
            temporal_type=temporal_type,
            backbone_name=backbone_name,
            num_segments=num_segments)
        cfg.model.cls_head = feature_head

    # ---------------------- multiple view ----------------------
    if not args.multi_view:
        # average features among multiple views
        cfg.model.cls_head['average_clips'] = 'score'
        if cfg.model.type == 'Recognizer3D':
            for idx, transform in enumerate(test_pipeline):
                if transform.type == 'SampleFrames':
                    test_pipeline[idx]['num_clips'] = 1
        for idx, transform in enumerate(test_pipeline):
            if transform.type == 'SampleFrames':
                test_pipeline[idx]['twice_sample'] = False
            # if transform.type in ['ThreeCrop', 'TenCrop']:
            if transform.type == 'TenCrop':
                test_pipeline[idx].type = 'CenterCrop'

    # -------------------- pipeline settings  --------------------
    # assign video list and video root
    if args.video_list is not None:
        cfg.test_dataloader.dataset.ann_file = args.video_list
    if args.video_root is not None:
        if cfg.test_dataloader.dataset.type == 'VideoDataset':
            cfg.test_dataloader.dataset.data_prefix = dict(
                video=args.video_root)
        elif cfg.test_dataloader.dataset.type == 'RawframeDataset':
            cfg.test_dataloader.dataset.data_prefix = dict(img=args.video_root)
    args.video_list = cfg.test_dataloader.dataset.ann_file
    args.video_root = cfg.test_dataloader.dataset.data_prefix
    # use UntrimmedSampleFrames for long video inference
    if args.long_video_mode:
        # preserve features of multiple clips
        cfg.model.cls_head['average_clips'] = None
        cfg.test_dataloader.batch_size = 1
        is_recognizer2d = (cfg.model.type == 'Recognizer2D')

        frame_interval = args.frame_interval
        for idx, transform in enumerate(test_pipeline):
            if transform.type == 'UntrimmedSampleFrames':
                clip_len = transform['clip_len']
                continue
            # replace SampleFrame by UntrimmedSampleFrames
            elif transform.type in ['SampleFrames', 'UniformSample']:
                assert args.clip_interval is not None, \
                    'please specify clip interval for long video inference'
                if is_recognizer2d:
                    # clip_len of UntrimmedSampleFrames is same as
                    # num_clips for 2D Recognizer.
                    clip_len = transform['num_clips']
                else:
                    clip_len = transform['clip_len']
                    if frame_interval is None:
                        # take frame_interval of SampleFrames as default
                        frame_interval = transform.get('frame_interval')
                assert frame_interval is not None, \
                    'please specify frame interval for long video ' \
                    'inference when use UniformSample or 2D Recognizer'

                sample_cfgs = dict(
                    type='UntrimmedSampleFrames',
                    clip_len=clip_len,
                    clip_interval=args.clip_interval,
                    frame_interval=frame_interval)
                test_pipeline[idx] = sample_cfgs
                continue
        # flow input will stack all frames
        if cfg.test_dataloader.dataset.get('modality') == 'Flow':
            clip_len = 1

        if is_recognizer2d:
            from mmaction.models import ActionDataPreprocessor
            from mmaction.registry import MODELS

            @MODELS.register_module()
            class LongVideoDataPreprocessor(ActionDataPreprocessor):
                """DataPreprocessor for 2D recognizer to infer on long video.

                Which would stack the num_clips to batch dimension, to preserve
                feature of each clip (no average among clips)
                """

                def __init__(self, num_frames=8, **kwargs) -> None:
                    super().__init__(**kwargs)
                    self.num_frames = num_frames

                def preprocess(self, inputs, data_samples, training=False):
                    batch_inputs, data_samples = super().preprocess(
                        inputs, data_samples, training)
                    # [N*M, T, C, H, W]
                    nclip_batch_inputs = batch_inputs.view(
                        (-1, self.num_frames) + batch_inputs.shape[2:])
                    # data_samples = data_samples * \
                    #     nclip_batch_inputs.shape[0]
                    return nclip_batch_inputs, data_samples

            preprocessor_cfg = cfg.model.data_preprocessor
            preprocessor_cfg.type = 'LongVideoDataPreprocessor'
            preprocessor_cfg['num_frames'] = clip_len
            print("Using Clip length: ", clip_len)

    # -------------------- Dump predictions --------------------
    args.dump = osp.join(args.output_prefix, 'total_feats.pkl')
    dump_metric = dict(type='DumpResults', out_file_path=args.dump)
    cfg.test_evaluator = [dump_metric]
    cfg.work_dir = osp.join(args.output_prefix, 'work_dir')

    return cfg


def split_feats(args):
    total_feats = load(args.dump)
    if args.dump_score:
        total_feats = [sample['pred_scores']['item'] for sample in total_feats]

    video_list = list_from_file(args.video_list)
    video_list = [line.split(' ')[0] for line in video_list]

    for video_name, feature in zip(video_list, total_feats):
        dump(feature, osp.join(args.output_prefix, video_name + '.pkl'))
    os.remove(args.dump)



class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super(ExportModel, self).__init__()
        self.model = model
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model(x)
        return self.gap(x).view(x.size(0), -1)
# Example usage
if __name__ == "__main__":
    import multiprocessing
    torch.set_num_threads(4)

    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher

    cfg.load_from = args.checkpoint


    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    # start testing

    model = runner.model
    model.to(device)
    model.eval()

    # root_dirs = [
    #     # First get videos from train_session_set and add the prefix '/data1/ghufran/validation_flow'
    #     # "/data1/ghufran/video_features/validation_Raft-flow/" + vid for vid in video_list["train_session_set"]
    #     # "/data1/ghufran/validation_flow_farn/" + vid for vid in video_list["train_session_set"]
    #     "/data1/ghufran/validation_flow/" + vid for vid in video_list["train_session_set"]


    # ]

    root_dirs = [
        # Then get videos from test_session_set and add the prefix '/data1/ghufran/test_flow'
        # "/data1/ghufran/video_features/validation_Raft-flow/" + vid for vid in video_list["test_session_set"]
        # "/data1/ghufran/test_flow_farn/" + vid for vid in video_list["test_session_set"]
        "/data1/ghufran/test_flow/" + vid for vid in video_list["test_session_set"]


    ]


    # model = ExportModel(model)
    model.to(device)
    model.eval()
    # # Export the model as onnx and exit
    # example_input = torch.randn(1, 1, 10, 224, 224).to(device)
    # torch.onnx.export(model, example_input, "resnet_rgb.onnx", export_params=True,
    #                   opset_version=12, do_constant_folding=True,
    #                   input_names=['input'], output_names=['output'])
    # exit()

    # First debug 
    # root_dirs = ['/data1/ghufran/test_flow_farn/video_test_0000897/']
    out_folder = '/data1/ghufran/helping_python_scripts/IMFE_val/'
    for folder in tqdm(root_dirs):
        process_folder(folder, model, out_folder)