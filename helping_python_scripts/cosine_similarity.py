import numpy as np
# first_path = '/data1/ghufran/THUMOS/flow_kinetics_resnet50_raft/video_test_0000292.npy'
first_path = '/data1/ghufran/THUMOS/flow_feat_farn/video_test_0000897.npy'
second_path = 'video_test_0000897.npy'

# Both contains N, 2048 features
first_features = np.load(first_path)
second_features = np.load(second_path)

print(first_features.shape)
print(second_features.shape)

# first_features = first_features[3]
# second_features = second_features[3]
# # Flatten the features

# first_features = first_features[0]
# second_features = second_features[0]
# first_features = first_features[:-1]
first_features = first_features.flatten()
second_features = second_features.flatten()

# Remove last element from second_features

def cosine_similarity(a, b):
    # Calculate the dot product of the two feature arrays
    dot_product = np.dot(a, b)
    
    # Calculate the L2 norm of the two feature arrays
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Calculate the cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    return similarity
# Calculate the cosine similarity between the two feature arrays
cosine_similarity = cosine_similarity(first_features, second_features)
print(f"Cosine Similarity: {cosine_similarity}")