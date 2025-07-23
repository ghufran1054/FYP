import yaml
import torch
from torch.utils.data import DataLoader
from dataloader_flow import THUMOSDatasetFLOW
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test_dataloader():
    # Example configuration - modify this to match your actual config
    cfg = {
        'root_path': '/data1/ghufran/video_features/validation_Raft-flow',
        'video_list_path': '/data1/ghufran/MiniROAD/data_info/video_list.json',
        'data_name': 'THUMOS',
        'annotation_type': 'target_perframe',
        'window_size': 128,  # Example window size
        'stride': 4,       # Example stride
        'num_classes': 22,  # Example number of classes
        'batch_size': 1 ,    # For testing
        'no_rgb': True,  # Set to True to test flow only
        'no_flow': False,  # Set to True to test rgb only
        'rgb_type': 'rgb_kinetics_resnet50',
        'flow_type': 'flow_kinetics_resnet50',
    }

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = THUMOSDatasetFLOW(cfg, mode='test', rootpath=cfg['root_path'])  # Test with train mode first
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Test iteration
    print("Starting test iteration...")
    for batch_idx, (rgb_inputs, flow_inputs, targets) in enumerate(dataloader):
        try:
            # Basic shape checks
            # Move to device
            rgb_inputs = flow_inputs.to(device)  # Explicitly move to GPU
            targets = targets.to(device)
            # assert rgb_inputs.dim() == 5, f"Expected 5D tensor (B,T,C,H,W), got {rgb_inputs.shape}"
            # assert targets.dim() == 3, f"Expected 3D tensor (B,T,C), got {targets.shape}"
            
            B, T, C, H, W = rgb_inputs.shape
            Bt, Tt, N = targets.shape
            assert T == Tt, f"Expected same temporal length for inputs and targets, got {T} and {Tt}"
            # assert C == 10, "Expected 3 color channels"
            # assert H == 224 and W == 224, "Expected 224x224 images"
            # assert T == cfg['window_size'], f"Expected temporal length {cfg['window_size']}, got {T}"
            
            print(f"Batch {batch_idx} successful - shapes:")
            print(f"  RGB inputs: {rgb_inputs.shape}")
            print(f"  Targets: {targets.shape}")
            
            # Optional: Visualize first frame of first video in batch
            if batch_idx == -1:
                import matplotlib.pyplot as plt
                first_frame = rgb_inputs[0, 0].permute(1, 2, 0).numpy()
                first_frame = first_frame * np.array(STD) + np.array(MEAN)  # Un-normalize
                first_frame = np.clip(first_frame, 0, 1)
                
                plt.figure()
                plt.imshow(first_frame)
                plt.title(f"First frame of first batch (video {dataset.inputs[0][0]})")
                plt.show()
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            raise

    print("\nTest completed successfully!")
    print(f"Total batches tested: {batch_idx + 1}")
    print(f"Total samples: {len(dataset)}")

if __name__ == '__main__':
    test_dataloader()