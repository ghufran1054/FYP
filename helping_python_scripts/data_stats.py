import os
import numpy as np
from collections import defaultdict

class_index = [
    "Background", "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot",
    "Diving", "FrisbeeCatch", "GolfSwing", "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault", "Shotput", 
    "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking", "Ambiguous"
]

def count_events(class_sequence):
    events = defaultdict(int)
    prev_class = None
    
    for cls in class_sequence:
        if cls != prev_class:  # New event starts
            events[class_index[cls]] += 1
        prev_class = cls
    
    return events

def process_npy_files(folder):
    stats = {}
    
    for file in os.listdir(folder):
        if file.endswith(".npy") and ('test' in file or 'validation' in file):
            file_path = os.path.join(folder, file)
            data = np.load(file_path)  # Shape (N, 22)

            # Number of frames
            num_frames = data.shape[0]
            
            # Get class labels per frame
            class_labels = np.argmax(data, axis=1)
            
            # Count unique classes and their occurrences
            unique_classes, counts = np.unique(class_labels, return_counts=True)
            # if first class is 0, then it is background, so remove it
            if unique_classes[0] == 0:
                unique_classes = unique_classes[1:]
                counts = counts[1:]
            class_counts = {class_index[cls]: count for cls, count in zip(unique_classes, counts)}
            
            # Count events per class
            events = count_events(class_labels)
            
            stats[file] = {
                'unique_classes': [class_index[cls] for cls in unique_classes],
                'class_counts': class_counts,
                'events': dict(events),
                'num_frames': num_frames
            }
    
    return stats

if __name__ == "__main__":
    folder_path = "/data1/ghufran/THUMOS/target_perframe"  # Update if needed
    stats = process_npy_files(folder_path)
    
    for file, data in stats.items():

        # Printing only those where unique classes are more than 1
        if len(data['unique_classes']) <= 1:
            continue
        print(f"File: {file} with {data['num_frames']} frames")
        print(f"  Unique Classes: {data['unique_classes']}")
        print(f"  Class Counts: {data['class_counts']}")
        print(f"  Events: {data['events']}")
        print("-")
