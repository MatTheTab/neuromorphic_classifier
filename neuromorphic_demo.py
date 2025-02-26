import streamlit as st
import torch
import random
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import tempfile
import os
import shutil
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from IPython.display import HTML
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
import torch.nn.functional as F
import asyncio

# Ensure there's an event loop running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))


BATCH_SIZE = 64
IMAGE_REPITIONS = 4
EPOCHS = 30
SPIKE_GRAD = surrogate.fast_sigmoid(slope=25) #
BETA = 0.5
NUM_STEPS = 10
GAIN = 0.5
THRESHOLD_LATENCY = 1e-4
TAU = 0.9
THRESHOLD_DELTA = 0.5
NUM_CLASSES = 10

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
dataiter = iter(trainloader)
images, labels = next(dataiter)

class Neuromorphic_Net(nn.Module):
    '''
    A spiking neural network (SNN) model using leaky integrate-and-fire (LIF) neurons.

    Parameters:
    input_shape (tuple): Shape of the input data (batch_size, channels, height, width).
    num_classes (int): Number of output classes for classification.

    Attributes:
    conv1 (nn.Conv2d): First convolutional layer.
    lif1 (snn.Leaky): First LIF layer.
    conv2 (nn.Conv2d): Second convolutional layer.
    lif2 (snn.Leaky): Second LIF layer.
    conv3 (nn.Conv2d): Third convolutional layer.
    lif3 (snn.Leaky): Third LIF layer.
    output (nn.Linear): Fully connected output layer.
    lif4 (snn.Leaky): Fourth LIF layer.

    Methods:
    get_covnolution_output(): Computes the flattened output size after convolutional layers.
    reset_states(): Resets the membrane potentials of LIF layers.
    forward(x): Performs a forward pass through the network.

    Returns:
    torch.Tensor: Output spike activity of the final layer.
    '''
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.in_channels = input_shape[1]
        self.conv1 = nn.Conv2d(self.in_channels, 4, 3)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.lif3 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.output = nn.Linear(self.get_covnolution_output(), self.num_classes)
        self.lif4 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)

    def get_covnolution_output(self):
        dummy_input = torch.rand(*self.input_shape)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        cur1 = F.max_pool2d(self.conv1(dummy_input), 2)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = F.max_pool2d(self.conv3(spk2), 2)
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3.view(BATCH_SIZE, -1).shape[-1]

    def reset_states(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem4 = self.lif4.init_leaky()

    def forward(self, x):
        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)
        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)
        cur3 = F.max_pool2d(self.conv3(spk2), 2)
        spk3, self.mem3 = self.lif3(cur3, self.mem3)
        output = self.output(spk3.view(spk3.shape[0], -1))
        spk4, self.mem4 = self.lif4(output, self.mem4)
        return spk4
    
def encode_rate(input):
    '''
    Encodes input data into spike trains using rate coding.

    Parameters:
    input (torch.Tensor): Input tensor to be encoded.

    Returns:
    torch.Tensor: Rate-encoded spike train.
    '''
    return spikegen.rate(input, num_steps=NUM_STEPS, gain=GAIN)

def encode_time(input):
    '''
    Encodes input data into spike trains using time-to-first-spike (latency) coding.

    Parameters:
    input (torch.Tensor): Input tensor to be encoded.

    Returns:
    torch.Tensor: Time-encoded spike train.
    '''
    return spikegen.latency(input, num_steps=NUM_STEPS, tau=TAU, threshold=THRESHOLD_LATENCY)

def encode_delta(input):
    '''
    Encodes input data using delta modulation, based on changes in rate-coded spikes.

    Parameters:
    input (torch.Tensor): Input tensor to be encoded.

    Returns:
    torch.Tensor: Delta-encoded spike train.
    '''
    return spikegen.delta(spikegen.rate(input, num_steps=NUM_STEPS, gain=GAIN), threshold=THRESHOLD_DELTA, off_spike=True)


# Load the models
model_rate = Neuromorphic_Net(images.shape, NUM_CLASSES)
model_delta = Neuromorphic_Net(images.shape, NUM_CLASSES)
model_time = Neuromorphic_Net(images.shape, NUM_CLASSES)

model_rate.load_state_dict(torch.load("models/neuromorphic_rate_model.pth", weights_only=False))
model_delta.load_state_dict(torch.load("models/neuromorphic_delta_model.pth", weights_only=False))
model_time.load_state_dict(torch.load("models/neuromorphic_temporal_model.pth", weights_only=False))

# Select a random image and label
def select_random_example():
    index = random.randint(0, len(testset) - 1)
    img, label = testset[index]
    return img, label

# Encode, visualize, and process the image
def encode_and_visualize(encoding_method):
    img, label = select_random_example()
    img = img.unsqueeze(0)  # Add batch dimension

    # Select encoding method and model
    if encoding_method == "Rate Encoding":
        spike_data = encode_rate(img)
        model = model_rate
    elif encoding_method == "Time-to-First-Spike Encoding":
        spike_data = encode_time(img)
        model = model_time
    elif encoding_method == "Delta Encoding":
        spike_data = encode_delta(img)
        model = model_delta

    # Visualize the spike train
    spike_data_sample = spike_data[:, 0, 0]
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)

    # Save animation as a video file
    #temp_video_path = os.path.join(tempfile.gettempdir(), "spike_animation.mp4")
    #anim.save(temp_video_path, writer="ffmpeg", fps=10)
    temp_video_path = os.path.join(tempfile.gettempdir(), "spike_animation.gif")
    anim.save(temp_video_path, writer="pillow", fps=10)
    plt.close(fig)
    output_video_path = temp_video_path

    # Convert to a format Streamlit can display
    #output_video_path = os.path.join(tempfile.gettempdir(), "converted_spike_animation.mp4")
    #shutil.copy(temp_video_path, output_video_path)


    # Get model output
    model.reset_states()
    outputs = []
    for step in spike_data:
        output_step = model(step)
        outputs.append(output_step)
    outputs = torch.stack(outputs)
    predicted_label = torch.argmax(outputs[-1], dim=1)

    return output_video_path, f"Predicted: {predicted_label.item()}", f"Expected: {label}"

# Streamlit UI
st.title("🧠 Spiking Neural Network Visualization")
st.write(
    "This app demonstrates how a Spiking Neural Network (SNN) processes images using different encoding methods. "
    "Select an encoding method, and the app will convert a random test image into a spike train, "
    "visualize it, and use a trained SNN to classify the image. The model's predicted class will be compared with the expected label."
)

# Dropdown for encoding selection
encoding_method = st.selectbox(
    "Choose Encoding Method", 
    ["Rate Encoding", "Time-to-First-Spike Encoding", "Delta Encoding"]
)

# Run when the user clicks "Generate"
if st.button("Generate and Classify"):
    with st.spinner("Processing..."):
        video_path, predicted_label, expected_label = encode_and_visualize(encoding_method)
    
    # Display video animation
    st.image(video_path)

    # Display results
    st.success(predicted_label)
    st.info(expected_label)
