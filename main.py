from vlm.vlm_inference import VLMInference
from retrieval.retrieval_memory import RetrievalMemory
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gc
import time

start_time = time.time()
# Initialize model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

vlm = VLMInference(model, processor)
retriever = RetrievalMemory("memory/situation_action.json")

# System prompt
system_message = """You are a Vision Language Model specialized in analyzing visual data from self-driving car cameras.
Your primary task is to detect and identify objects in images captured by multiple cameras mounted on the self-driving car, which provides a first-person perspective and recommend appropriate driving actions.
You will be provided with two images:
    1. Front View Camera: Showing the front view of the self-driving car.
    2. Back-View Camera: Showing the rear view of the self-driving car.

Focus on accuracy, clarity, and safety-critical information. Avoid unnecessary explanations."""

question_1_text = "The first image comes from a self-driving car\’s front view camera. The second image is from the car\’s back camera, \
             showing the view behind the vehicle, with vehicles that are following the self-driving car. \
             The view captured by the back camera is a direct representation of the rear surroundings, with the self-driving car in the center of the scene.\
             Question 1: Can you describe the key elements in the front and back views, such as road conditions, objects, vehicles, and potential hazards?\
             Please differentiate between the two views in your description. Keep your answer short."

# Initial message + question 1
messages = [
    {"role": "system", "content": [{"type": "text", "text": system_message}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./VAD_model/3378/rgb_front/0013.png"}, # 11 12 13!
            {"type": "image", "image": "./VAD_model/3378/rgb_back/0013.png"},
            {"type": "text", "text": question_1_text}
        ],
    }
]

# Format input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to("cuda")

# Run inference for qn 1
generated_ids = vlm.generate(inputs)
output_q1 = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
print("Answer Q1:", output_q1)

question_2_text = "Question 2: Based on question 1, summarise the situation in a short phrase only."

# Question 2
messages.append({"role": "assistant", "content": [{"type": "text", "text": question_1_text + "\n" + output_q1}]})
messages.append({
    "role": "user",
    "content": [{"type": "text", "text": question_2_text}]
})
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to("cuda")

generated_ids = vlm.generate(inputs, max_new_tokens=512)
output_q2 = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
print("Answer Q2:", output_q2)

# save qn2 embedding and run cosine similarity
qwen_embedding = vlm.get_generation_embedding(inputs, generated_ids)

# Question 3
messages.append({"role": "assistant", "content": [{"type": "text", "text": output_q2}]})
messages.append({
    "role": "user",
    "content": [{"type": "text", "text": "Question 3: Based on your description, what actions should the self-driving car take to ensure safety?\n \
             Consider the current lane position, road conditions, and potential hazards in your response.\n \
             Should the car change lanes, adjust speed, or take any other specific actions to prioritize safety?\n \
             Please summarise your answer for question 3 in one sentence. \
                 Furthermore, The current position of the self-driving car is (0.0, 0.0). The next 4 waypoints of the self-driving car\'s path are: (0.0, 3.4), (0.0, 7.2), (-0.1, 11.0), (-0.1, 15.0). \
             What is the path taken by the self-driving car? Is this path taken by the self-driving car align with your suggested action?"}]
})

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to("cuda")

generated_ids = vlm.generate(inputs, max_new_tokens=512)
output_q3 = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
print("Answer Q3:", output_q3)

# check if question 3 answer has "not aligned" inside
if 'not aligned' in output_q3.lower():
    # check "memory module" in json file to determine if sim score >= 0.5 then extract action
    results = retriever.query(qwen_embedding)
    print(results[0]['similarity'])
    if results[0]['similarity'] < 0.8:
        print("No good match found.")
        # add current situation and action to json file to update memory
        retriever.save_new_situation(
            embedding_tensor=qwen_embedding,
            situation_text=output_q2,
            full_action_text=output_q3
        )
    else:
        print("top match:", results[0])
        print("retrieved action:", results[0]["action"])
else:
    # do nothing
    print('Nothing to be done, action is aligned.')
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
torch.cuda.empty_cache()

