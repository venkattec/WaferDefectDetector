from calculateDefect import detect_defects
from localizeDefect import detect_and_localize_defects
from visualTransformer import find_defects
from langchain.tools import Tool, StructuredTool
from PIL import Image
import torch
import cv2
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from langchain_community.llms import Ollama
from pydantic import BaseModel
import numpy as np

#Multimodel tool llava
class MultimodalInput(BaseModel):
    image_path: str
    input: str


class MultimodalLLM:
    def __init__(self):
        # Initialize processor and model
        self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

    def analyze(self, image_path: str, input: str):
        try:
            # Load and preprocess the image
            image = Image.open(image_path)
            prompt = f"USER: <image>\nConsider yourself as wafer defect detector in a semiconductor{input}. ASSISTANT:"
            print(prompt)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

            # Debug: Check image tokenization
            print("Pixel values shape:", inputs.get("pixel_values", None).shape)

            # Generate output
            with torch.no_grad():
                # outputs = self.model.generate(**inputs)
                generate_ids = self.model.generate(**inputs, max_new_tokens=100)
            return \
            self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # Decode and return the response
            # return self.processor.decode(outputs[0], skip_special_tokens=True)


        except Exception as e:
            print(f"Error during analysis: {e}")
            return "An error occurred during processing."


class LocalizeDefectInput(BaseModel):
    image_path: str  # Path to the wafer map image
    input: str  # Question to answer, e.g., "What percentage of the wafer is defective?"


def localize_defect(image_path: str, input: str):
    if not image_path or not input:
        return "Error: Missing 'image_path' or 'question' in the input."
    if not os.path.exists(image_path):
        return f"Error: File at '{image_path}' does not exist."
    print(image_path, input)
    output = detect_and_localize_defects(image_path=image_path)
    return {"output": output}


class DefectPercentageInput(BaseModel):
    image_path: str  # Path to the wafer map image
    input: str  # Question to answer, e.g., "What percentage of the wafer is defective?"


def calculate_defect_percentage(image_path: str, question: str):
    print(f"Received image_path: {image_path}")  # Debugging input
    try:
        if not image_path:
            return "Error: Image path is missing."
        defect_percentage = detect_defects(image_path=image_path)
        # print(f"Defect Percentage: {defect_percentage:.2f}%")  # Debugging log
        return {'output': str(defect_percentage)}
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging log
        return f"Error while calculating defect percentage: {str(e)}"


class ClassifyDefectInput(BaseModel):
    image_path: str  # Path to the wafer map image


def classify_defect(image_path: str):
    print(f"Received image_path: {image_path}")
    return {'output' : 'done'}


# Initialize the MultimodalLLM instance
# multimodal_llm = MultimodalLLM()
#
# # Define the structured tool
# multimodal_tool = StructuredTool.from_function(
#     func=multimodal_llm.analyze,
#     name="MultimodalLLMTool",
#     input_schema=MultimodalInput,
#     description="Analyze wafer map images and answer questions about them.",
# )
# defect_classification_tool = StructuredTool.from_function(
#     func=classify_defect,
#     input_schema=ClassifyDefectInput,
#     name="DefectClassifier",
#     description="Use this tool to classify defects in the wafer map image.",
#     return_direct=True
# )
#
# defect_localize_tool = StructuredTool.from_function(
#     func=localize_defect,
#     input_schema=LocalizeDefectInput,
#     name="DefectLocalizer",
#     description="Use this tool to localize defects in the wafer map image.",
#     return_direct=True
# )

defect_percentage_tool = StructuredTool.from_function(
    func=calculate_defect_percentage,
    input_schema=DefectPercentageInput,
    name="DefectPercentageCalculator",
    description="Use this tool to calculate the percentage of defect area in a wafer map image.",
    return_direct=True
)
