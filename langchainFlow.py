from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.chains import LLMChain, TransformChain, SequentialChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import os
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from calculateDefect import detect_defects
from localizeDefect import detect_and_localize_defects
from visualTransformer import find_defects
from save_img import cnv
import numpy as np
import torch
from PIL import Image


llava_processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")


# Define tools
@tool
def multimodal_tool(image_path: str, question: str) -> str:
    """
    Performs a multimodal analysis on the given image and provides a general summary.
    """
    print(f"Received image_path in llava: {image_path}")  # Debugging input
    try:
        # Load and preprocess the image
        image_path = cnv(image_path)
        image = Image.open(image_path)
        prompt = f"USER: <image>\nConsider yourself as a image analyst and please answer the question from the user based on the image given.User will upload a wafer image from the semiconductor factory. Answer the question based on that wafer image {question}. ASSISTANT:"
        # print(prompt)
        inputs = llava_processor(images=image, text=prompt, return_tensors="pt")

        # Debug: Check image tokenization
        # print("Pixel values shape:", inputs.get("pixel_values", None).shape)

        # Generate output
        with torch.no_grad():
            # outputs = self.model.generate(**inputs)
            generate_ids = llava_model.generate(**inputs, max_new_tokens=100)
            answer = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return answer.split('ASSISTANT:')[-1]
        # Decode and return the response
        # return self.processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return "An error occurred during processing."


@tool
def defect_percentage_tool(image_path: str, question: str) -> str:
    """
    Calculates the percentage of the wafer that is defective.
    """
    print(f"Received image_path: {image_path}")  # Debugging input
    try:
        if not image_path:
            return "Error: Image path is missing."
        defect_percentage = detect_defects(npy_path=image_path)
        # print(f"Defect Percentage: {defect_percentage:.2f}%")  # Debugging log
        return str(defect_percentage)
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging log
        return f"Error while calculating defect percentage: {str(e)}"


@tool
def defect_localize_tool(image_path: str, question: str) -> str:
    """
    Localizes the largest defect in the wafer image.
    """
    print(f"Received image_path: {image_path}")  # Debugging input
    if not image_path or not input:
        return "Error: Missing 'image_path' or 'question' in the input."
    if not os.path.exists(image_path):
        return f"Error: File at '{image_path}' does not exist."

    output = detect_and_localize_defects(npy_path=image_path)
    return str(output)


@tool
def defect_classification_tool(image_path: str, question: str) -> str:
    """
    Classifies the defect type in the wafer image.
    """
    save_image = cnv(image_path)
    print(f"Received image_path: {image_path}")  # Debugging input
    try:
        array = np.load(image_path, allow_pickle=True)
        array = np.expand_dims(array, -1)  # Add batch dimension
        image = np.array([array])
        output = find_defects(image)
        return output
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging log
        return str(e)


# Define output schema
class DefectAnalysisOutput(BaseModel):
    result: str = Field(..., description="The output of the selected tool based on the user's question.")


parser = PydanticOutputParser(pydantic_object=DefectAnalysisOutput)

# Initialize LLM
llm = Ollama(model="llama3")

# Define prompt template
tool_selection_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an assistant analyzing wafer maps. The human has asked a question.

    Question: {question}

    You have access to the following tools:
    1. DefectPercentageCalculator: Calculates the percentage of the wafer that is defective.
    2. DefectLocalizer: Localizes the largest defect on the wafer.
    3. DefectClassifier: Classifies the defect type in the wafer image.
    4. MultimodalLLMTool: Answers general questions about the wafer image.

    Based on the question, select the most relevant tool and return its name as:
    Tool: <tool_name>
    """
)

# Tool selection chain
tool_selector = LLMChain(
    llm=llm,
    prompt=tool_selection_prompt,
    output_key="selected_tool"
)


# Tool execution logic
def execute_tool(selected_tool: str, image_path: str, question: str):
    """
    Executes the selected tool and returns the result.
    """
    tool_mapping = {
        "DefectPercentageCalculator": defect_percentage_tool,
        "DefectLocalizer": defect_localize_tool,
        "DefectClassifier": defect_classification_tool,
        "MultimodalLLMTool": multimodal_tool,
    }
    selected_tool = selected_tool.split(':')[-1].strip()
    print("selected tool: ",selected_tool)
    if selected_tool in tool_mapping:
        tool = tool_mapping[selected_tool]

        # Ensure inputs are passed as a dictionary with correct types
        inputs = {"image_path": str(image_path), "question": str(question)}
        return tool(inputs)  # Pass dictionary to the tool
    else:
        return "Invalid tool selected."



# TransformChain for tool execution
tool_executor = TransformChain(
    input_variables=["selected_tool", "image_path", "question"],
    output_variables=["tool_result"],
    transform=lambda inputs: {
        "tool_result": execute_tool(inputs["selected_tool"], inputs["image_path"], inputs["question"])
    }
)

# Create a SequentialChain to combine selection and execution
chain = SequentialChain(
    chains=[tool_selector, tool_executor],
    input_variables=["image_path", "question"],
    output_variables=["tool_result"]
)

def get_answer(question,file_path):
    # Inputs
    inputs = {
        "image_path": os.path.abspath(file_path),
        "question": question
    }

    # Run the chain
    response = chain.run(inputs)
    print("Agent Response:", response)
    return response
