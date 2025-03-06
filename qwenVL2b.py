from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

### Util function to clear cuda memory
import gc
import time
def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
### -----------------------------------------------------------------------------------------------


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# system message
system_message = """You are a Vision Language Model specialized in analyzing visual data from self-driving car cameras.
Your primary task is to detect and identify objects in images captured by multiple cameras mounted on the self-driving car, which provides a first-person perspective and recommend appropriate driving actions.
You will be provided with two images:
    1. Front View Camera: Showing the front view of the self-driving car.
    2. Back-View Camera: Showing the rear view of the self-driving car.

Tasks:
    1. Object Detection: Identify and label safety-critical objects (e.g., pedestrians, vehicles, traffic signs, lanes, obstacles) in both images.
    - Pay special attention to **emergency vehicles** (e.g., ambulances, fire trucks, police cars). These vehicles are high-priority objects. These vehicles are usually indicated by flashing lights on the vehicle.
    2. Driving Action Recommendation: Based on the detected objects and their positions, output the recommended driving action (e.g., 'stop', 'change lane', 'proceed cautiously', 'yield').

Respond with concise and clear answers:
    Object detection results should be short phrases (e.g., 'red traffic light', '2 pedestrians on the right', 'blue car approaching from behind').
    Driving actions should be a single phrase summarizing the appropriate maneuver.

Focus on accuracy, clarity, and safety-critical information. Avoid unnecessary explanations."""

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_message}],
    },
    {
        "role": "user",
        "content": [
            # {"type": "image", "image": "./3373_22front.png"},
            # {"type": "image", "image": "./3373_22back.png"},
            {"type": "image", "image": "./rgb_front/0025.png"},
            {"type": "image", "image": "./rgb_back/0025.png"},
            {"type": "text", "text": "The first image comes from a self-driving car\’s front view camera. The second image is from the car\’s back camera, showing the view behind the vehicle, with vehicles that are following the self-driving car. The view captured by the back camera is a direct representation of the rear surroundings, with the self-driving car in the center of the scene.\
             Question 1: Can you describe the key elements in the front and back views, such as road conditions, objects, vehicles, and potential hazards? Please differentiate between the two views in your description.\
             Question 2: Based on your description, what actions should the self-driving car take to ensure safety? Consider the current lane position, road conditions, and potential hazards in your response. Should the car change lanes, adjust speed, or take any other specific actions to prioritize safety?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=1280)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Output text is in a list of length 1: print list[0] for output
print(output_text[0])

# Clear cuda memory
clear_memory()