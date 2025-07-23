import tensorrt as trt
import numpy as np
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import time
from threading import Thread, Lock
from collections import deque
import onnxruntime as ort

mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # Mean values for R, G, B channels
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

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
def convert_flow_to_image(flow_x, flow_y, lower_bound=-20, higher_bound=20):
    """
    Converts optical flow components to 8-bit image representations.

    Args:
        flow_x (np.ndarray): Optical flow in x-direction (float32).
        flow_y (np.ndarray): Optical flow in y-direction (float32).
        lower_bound (float): Minimum value to clamp/scale.
        higher_bound (float): Maximum value to clamp/scale.

    Returns:
        img_x (np.ndarray): Scaled x-flow image (uint8).
        img_y (np.ndarray): Scaled y-flow image (uint8).
    """
    def cast(v):
        return np.clip(np.round(255.0 * (v - lower_bound) / (higher_bound - lower_bound)), 0, 255).astype(np.uint8)

    img_x = cast(flow_x)
    img_y = cast(flow_y)
    return img_x, img_y

opening_time = 0

def resize(img):
    # Image size is 455x256, take a center crop of 224x224
    height, width, _ = img.shape
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = left + 224
    bottom = top + 224

    # Crop the image
    img = img[top:bottom, left:right]
    return img

def resize_flow(img):
    """img: (2, 256, 455)"""
    _, height, width = img.shape
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = left + 224
    bottom = top + 224

    # Crop the image
    img = img[:, top:bottom, left:right]
    return img

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

                
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,inputs,batch_size=1):
        
        for i,x in enumerate(inputs):
            x = x.astype(self.dtype)
            np.copyto(self.inputs[i].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]

class OpticalFlow:
    def __init__(self, method='opencv'):
        # method can be 'opencv (farneback)' and RAFT (trt model)
        self.method = method

        # If its RAFT trt model, setup the model
        if self.method == 'raft':
            self.model = TrtModel('raft.trt')
        
        if self.method == 'onnx':
            self.model = ort.InferenceSession('raft_model.onnx')
            self.input_name1 = self.model.get_inputs()[0].name
            self.input_name2 = self.model.get_inputs()[1].name
            self.output_name1 = self.model.get_outputs()[0].name


    def compute(self, img1, img2):
        if self.method == 'opencv':
            # Convert images to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Compute optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 13, 3, 5, 1.1, 0)

            return convert_flow_to_image(flow[..., 0], flow[..., 1])  # Return flow in x and y directions

        elif self.method == 'raft':
            input1_shape = self.model.engine.get_binding_shape(0)
            input2_shape = self.model.engine.get_binding_shape(1)

            # Assuming img1 and img2 are already resized to the input shape
            flow = self.model([img1, img2])[0]
            # We get flattened output, so we need to reshape it to 2,224,224
            flow = np.reshape(flow, (2, 224, 224))
            flow_x = flow[0]
            flow_y = flow[1]
            return flow_x, flow_y
        elif self.method == 'onnx':
            # Add batch dimension
            # Transpose the image to match the input shape expected by the model (C, H, W)
            img1 = np.transpose(img1, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img2 = np.transpose(img2, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img1 = np.expand_dims(img1, axis=0)
            img2 = np.expand_dims(img2, axis=0)
            flow = self.model.run([self.output_name1], {self.input_name1: img1, self.input_name2: img2})[0]
            # Remove the batch dimension
            flow = np.squeeze(flow)
            flow_x = flow[0]
            flow_y = flow[1]
            return flow_x, flow_y
            # Remove the batch dimension
        else:
            return None, None
        

class FrameReader:
    def __init__(self, method='live', frames_path=None, src=0, only_rgb=False):

        # This class reads frames from source and groups them into a batch of 6
        if method not in ['live', 'video']:
            raise ValueError("Method must be 'live' or 'video'")
        if method == 'video' and frames_path is None:
            raise ValueError("video_path must be provided for video method")
        self.method = method
        self.frames_path = frames_path
        self.frame_count = 0

        self.frame_buffer = deque(maxlen=6)
        self.flow_buffer = deque(maxlen=5) if only_rgb == False else None
        self.batch_size = 6

        self.frame_lock = Lock()
        self.flow_lock = Lock()
        self.exit = False

        if self.method == 'live':
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
        
        elif self.method == 'video':
            self.frames = os.listdir(self.frames_path)
            self.frames.sort()
            self.frames = self.frames[0:1000] # DEBUG
            self.frame_count = len(self.frames)
            self.current_frame_index = 0
        
    def release(self):
        if self.method == 'live':
            self.cap.release()
    
    def start(self):
        if self.method == 'live':
            # start the thread to read frames from the video stream
            Thread(target=self.update, args=()).start()
            return self

    def read(self):
        if self.exit == True:
            return "exit", "exit"
        
        if self.method == 'live':
            if len(self.frame_buffer) == self.batch_size:
                if self.flow_buffer is not None and len(self.flow_buffer) != self.batch_size - 1:
                    return None, None
                with self.frame_lock:
                    batch = list(self.frame_buffer)
                with self.flow_lock:
                    flow_batch = list(self.flow_buffer) if self.flow_buffer is not None else None
                self.frame_count += self.batch_size
                
                # Returning only one frame and five optical flow frames
                img = batch[4]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Performing Normalization
                img = (img - mean) / std
                img = np.transpose(img, (2,0,1))  # (H, W, C) -> (C, H, W)
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=0)
                return img, flow_batch if flow_batch is not None else None
            else:
                return None, None

        elif self.method == 'video':
            # Load 6 frames at a time
            if self.current_frame_index + self.batch_size <= self.frame_count:
                batch = []
                for i in range(self.batch_size):
                    frame_path = os.path.join(self.frames_path, self.frames[self.current_frame_index + i])
                    start_time = time.time()
                    img = cv2.imread(frame_path)
                    img = resize(img)
                    end_time = time.time()

                    global opening_time
                    opening_time += end_time - start_time
                    batch.append(img)

                    # Compute optical flow for the last two frames
                    if i > 0 and self.flow_buffer is not None:
                        img1 = batch[i - 1]
                        img2 = batch[i]
                        flow_x, flow_y = optical_flow.compute(img1, img2)
                        # Concatenate them in the channel dimension to form a 2 channel image
                        flow_x = np.expand_dims(flow_x, axis=0)
                        flow_y = np.expand_dims(flow_y, axis=0)
                        flow = np.concatenate((flow_x, flow_y), axis=0)
                        self.flow_buffer.append(flow)

                self.current_frame_index += self.batch_size
            
                # Returning only one frame and five optical flow frames
                img = batch[4]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = (img - mean) / std
                img = np.transpose(img, (2,0,1))  # (H, W, C) -> (C, H, W)
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=0)
                return img, list(self.flow_buffer) if self.flow_buffer is not None else None
            else:
                return "exit", "exit"

    def update(self):
        while True:
            if self.method == 'live':
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from webcam")
                    break
                frame = cv2.resize(frame, (455, 256))
                frame = resize(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.exit = True

                with self.frame_lock:
                    self.frame_buffer.append(frame)
                
                # If there are two frames in the buffer we can compute the optical flow between the most recent two frames
                if len(self.frame_buffer) > 1 and self.flow_buffer is not None:
                    with self.frame_lock:
                        img1 = self.frame_buffer[-2]
                        img2 = self.frame_buffer[-1]
                    flow_x, flow_y = optical_flow.compute(img1, img2)
                    # Concatenate them in the channel dimension to form a 2 channel image
                    flow_x = np.expand_dims(flow_x, axis=0)
                    flow_y = np.expand_dims(flow_y, axis=0)
                    flow = np.concatenate((flow_x, flow_y), axis=0)

                    with self.flow_lock:
                        self.flow_buffer.append(flow)



if __name__ == "__main__":
 
    batch_size = 1
    only_rgb = True
    trt_engine_path_resnet_rgb = os.path.join("mobileone.trt")
    trt_engine_path_resnet_flow = os.path.join("resnet_flow.trt")
    if only_rgb:
        trt_engine_path_mroad = os.path.join("mroad_rgb_mob.trt")
    else:
        trt_engine_path_mroad = os.path.join("mroad_flow_farn_mob.trt")
    model_mroad = TrtModel(trt_engine_path_mroad)
    model_resnet_rgb = TrtModel(trt_engine_path_resnet_rgb)
    model_resnet_flow = TrtModel(trt_engine_path_resnet_flow)
    optical_flow = OpticalFlow(method='opencv')

    rgb_feat_shape = model_mroad.engine.get_binding_shape(0)
    flow_feat_shape = model_mroad.engine.get_binding_shape(1)
    hidden_shape = model_mroad.engine.get_binding_shape(2)

    rgb_input_shape = model_resnet_rgb.engine.get_binding_shape(0)
    flow_input_shape = model_resnet_flow.engine.get_binding_shape(0)

    video = 'video_test_0000179'
    video = 'video_validation_0000202'
    mode = 'live'
    cam_src = 1
    frame_reader = FrameReader(method=mode, frames_path=os.path.join(video), src=cam_src, only_rgb=only_rgb)
    frame_reader.start()
    prediction = []
    i = 0
    hidden = np.zeros((1, 1, 1024))
    


    start_time = time.time()
    while True:
        img, flow_list = frame_reader.read()
        if type(img) == str and img == "exit":
            break
        if img is None:
            continue
        
        # We have one image and we have 5 (x,y) flow frames
        # One image will be output in RGB feature extractor to get rgb_feat
        # 5 (x,y) frames will be input inside Flow feature extractor

        # Lets first print the shape of each
        # Concatenate the all 5 (2, 224, 224) flow frames to get (10, 224, 224)
        if not only_rgb:
            flow = np.concatenate(flow_list, axis=0)
            flow = np.expand_dims(flow, axis=0)
            flow = np.expand_dims(flow, axis=0)


        # First send image in ResNet RGB feature extractor
        rgb_feat = model_resnet_rgb(img)[0]
        rgb_feat = np.expand_dims(rgb_feat, axis=0)

        # Then pass flow through ResNet Flow feature extractor
        if not only_rgb:
            flow_feat = model_resnet_flow([flow])[0]
            flow_feat = np.expand_dims(flow_feat, axis=0)
        # Now we have rgb_feat and flow_feat, we can pass them to mroad model
        # We need to pass hidden state as well
        if only_rgb:
            output = model_mroad([rgb_feat, hidden])
        else:
            output = model_mroad([rgb_feat, flow_feat, hidden])
        pred = output[1]
        hidden = output[0]
        # print(pred.shape)
        idx = np.argmax(pred[0])
        if idx != 0:
            print(f"Event Detected: {all_class_names[idx]} with probability {pred[0][idx]:.2f}")
        else:
            print(f"No Event with probability {pred[0][idx]:.2f}       ", end='\r')


    end_time = time.time()

    print("Processed ", frame_reader.frame_count , " frames in ", end_time - start_time, " s")
    print("FPS: ", frame_reader.frame_count / (end_time - start_time))


    print("Time excluding opening images: ", end_time - start_time - opening_time)
    print("FPS excluding opening images: ", frame_reader.frame_count / (end_time - start_time - opening_time))





    # np.save('pred_scores.npy',np.array(pred_scores))
