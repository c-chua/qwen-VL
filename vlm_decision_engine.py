import torch
from vlm.vlm_inference import VLMInference
from retrieval.retrieval_memory import RetrievalMemory
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from agents.navigation.local_planner import RoadOption  # From car project

class VLMDecisionEngine:
    def __init__(self, memory_path="memory/situation_action.json", device="cuda"):
        self.device = device
        self.memory_path = memory_path

        # Load VLM model + processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        # Wrap with your inference and memory classes
        self.vlm = VLMInference(self.model, self.processor)
        self.retriever = RetrievalMemory(memory_path, device=device)

    def run_inference_from_images(self, front_img_path, back_img_path):
        # Build messages
        system_message = """You are a Vision Language Model specialized in analyzing visual data from self-driving car cameras.
        Your primary task is to detect and identify objects in images captured by multiple cameras mounted on the self-driving car, which provides a first-person perspective and recommend appropriate driving actions.
        You will be provided with two images:
            1. Front View Camera: Showing the front view of the self-driving car.
            2. Back-View Camera: Showing the rear view of the self-driving car.

        Focus on accuracy, clarity, and safety-critical information. Avoid unnecessary explanations."""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": front_img_path},
                    {"type": "image", "image": back_img_path},
                    {"type": "text", "text": "The first image comes from a self-driving car\’s front view camera. The second image is from the car\’s back camera, \
                    showing the view behind the vehicle, with vehicles that are following the self-driving car. \
                    The view captured by the back camera is a direct representation of the rear surroundings, with the self-driving car in the center of the scene.\
                    Question 1: Can you describe the key elements in the front and back views, such as road conditions, objects, vehicles, and potential hazards?\
                    Please differentiate between the two views in your description. Keep your answer short."}
                ],
            }
        ]

        # Format input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.device)

        # Run Q1
        gen_ids = self.vlm.generate(inputs)
        q1_output = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        print(q1_output)

        # Add Q2
        messages.append({"role": "assistant", "content": [{"type": "text", "text": q1_output}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "Question 2: Based on question 1, summarise the situation in a short phrase only."}]})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        gen_ids = self.vlm.generate(inputs)
        q2_output = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        q2_embedding = self.vlm.get_generation_embedding(inputs, gen_ids)
        print(q2_output)

        # Add Q3
        messages.append({"role": "assistant", "content": [{"type": "text", "text": q2_output}]})
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Based on your description, what actions should the self-driving car take to ensure safety?\n \
                    Consider the current lane position, road conditions, and potential hazards in your response.\n \
                    Should the car change lanes, adjust speed, or take any other specific actions to prioritize safety?\n \
                    Please summarise your answer for question 3 in one sentence. \
                        Furthermore, The current position of the self-driving car is (0.0, 0.0). The next 4 waypoints of the self-driving car\'s path are: (0.0, 3.4), (0.0, 7.2), (-0.1, 11.0), (-0.1, 15.0). \
                    What is the path taken by the self-driving car? Is this path taken by the self-driving car align with your suggested action?"}]
        })
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        gen_ids = self.vlm.generate(inputs)
        q3_output = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        print(q3_output)

        if "not aligned" in q3_output.lower():
            results = self.retriever.query(q2_embedding)
            if results and results[0]['similarity'] >= 0.5:
                action = results[0]["action"]
                override_command = self.map_vlm_action_to_road_option(action)
                print(f"[VLM] Overriding command to: {override_command} (Action: {action})")
                return override_command
            else:
                print("[VLM] No good match found in memory.")
        else:
            print("[VLM] Situation aligned. No override needed.")

        return None

    def map_vlm_action_to_road_option(self, action: str):
        action = action.lower()
        if "change lane right" in action:
            return RoadOption.CHANGELANERIGHT
        elif "change lane left" in action:
            return RoadOption.CHANGELANELEFT
        elif "turn left" in action:
            return RoadOption.LEFT
        elif "turn right" in action:
            return RoadOption.RIGHT
        elif "continue" in action or "go straight" in action:
            return RoadOption.LANEFOLLOW
        return RoadOption.LANEFOLLOW
    
    def run_inference_from_pil(self, front_img, back_img):
        image_inputs = [
            front_img.resize((336, 336)),
            back_img.resize((336, 336))
        ]

        # Build messages
        system_message = """You are a Vision Language Model specialized in analyzing visual data from self-driving car cameras.
        Your primary task is to detect and identify objects in images captured by multiple cameras mounted on the self-driving car, which provides a first-person perspective and recommend appropriate driving actions.
        You will be provided with two images:
            1. Front View Camera: Showing the front view of the self-driving car.
            2. Back-View Camera: Showing the rear view of the self-driving car.

        Focus on accuracy, clarity, and safety-critical information. Avoid unnecessary explanations."""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": front_img},
                    {"type": "image", "image": back_img},
                    {"type": "text", "text": "The first image comes from a self-driving car\’s front view camera. The second image is from the car\’s back camera, \
                    showing the view behind the vehicle, with vehicles that are following the self-driving car. \
                    The view captured by the back camera is a direct representation of the rear surroundings, with the self-driving car in the center of the scene.\
                    Question 1: Can you describe the key elements in the front and back views, such as road conditions, objects, vehicles, and potential hazards?\
                    Please differentiate between the two views in your description. Keep your answer short."}
                ],
            }
        ]

        # Format input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.device)

        # Run Q1
        gen_ids = self.vlm.generate(inputs)
        q1_output = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        print(q1_output)

        # Add Q2
        messages.append({"role": "assistant", "content": [{"type": "text", "text": q1_output}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "Question 2: Based on question 1, summarise the situation in a short phrase only."}]})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        gen_ids = self.vlm.generate(inputs)
        q2_output = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        q2_embedding = self.vlm.get_generation_embedding(inputs, gen_ids)
        print(q2_output)

        # Add Q3
        messages.append({"role": "assistant", "content": [{"type": "text", "text": q2_output}]})
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Based on your description, what actions should the self-driving car take to ensure safety?\n \
                    Consider the current lane position, road conditions, and potential hazards in your response.\n \
                    Should the car change lanes, adjust speed, or take any other specific actions to prioritize safety?\n \
                    Please summarise your answer for question 3 in one sentence. \
                        Furthermore, The current position of the self-driving car is (0.0, 0.0). The next 4 waypoints of the self-driving car\'s path are: (0.0, 3.4), (0.0, 7.2), (-0.1, 11.0), (-0.1, 15.0). \
                    What is the path taken by the self-driving car? Is this path taken by the self-driving car align with your suggested action?"}]
        })
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        gen_ids = self.vlm.generate(inputs)
        q3_output = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        print(q3_output)

        if "not aligned" in q3_output.lower():
            results = self.retriever.query(q2_embedding)
            if results and results[0]['similarity'] >= 0.5:
                action = results[0]["action"]
                override_command = self.map_vlm_action_to_road_option(action)
                print(f"[VLM] Overriding command to: {override_command} (Action: {action})")
                return override_command
            else:
                print("[VLM] No good match found in memory.")
        else:
            print("[VLM] Situation aligned. No override needed.")

        return None