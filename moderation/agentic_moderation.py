import json
import os
import yaml
import random
import time

from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

from distilling_agentic_moderation import run_inference_with_model,load_model_for_inference


class ReviewModerationAgent:
    def __init__(self, llm, system_description, task, goal, decision_options, allow_distilled_model=False, distilled_model_accuracy_threshold=90):
        """
        Initialize the moderation agent with an LLM and relevant task context.
        """
        self.llm = llm  # LangChain's LLM object
        self.system_description = system_description  # High-level description of the system
        self.task = task  # Task description (e.g., "Moderate product reviews")
        self.goal = goal  # Goal description (e.g., "Determine if a review is offensive")
        self.decision_options = decision_options  # Options for decision-making (e.g., ["offensive", "appropriate"])
        self.allow_distilled_model = allow_distilled_model  # Enable or disable the use of distilled models
        self.distilled_model_accuracy_threshold = distilled_model_accuracy_threshold  # Accuracy threshold for distilled models
        self.input = None  # To save the input review being processed
        self.output = None  # To save the LLM's response

        try:
            self.distilled_model_path = "./moderation_results/distilled_model.pth"
            self.details_path = "./moderation_results/distilled_model_details.json"
            self.distilled_model,self.distilled_device = load_model_for_inference(self.distilled_model_path)
        except:
            print('error loading distilled model')

    def moderate_review(self, review_text, context=""):
        """
        Use the LLM or distilled model to decide if a review is offensive or appropriate.
        """
        self.input = review_text

        # Placeholder: Check if using distilled model is allowed
        if self.allow_distilled_model:

            # Check if distilled model and details file exist
            if os.path.exists(self.distilled_model_path) and os.path.exists(self.details_path):
                with open(self.details_path, "r") as f:
                    details = json.load(f)

                # Check if the distilled model accuracy meets the threshold
                if details.get("final_accuracy_%", 0) >= self.distilled_model_accuracy_threshold:
                    print('Note: agent using distilled model for moderation')
                    # Load the label map
                    label_map = details.get("label_map", {})
                    reverse_label_map = {v: k for k, v in label_map.items()}

                    predicted = run_inference_with_model(self.distilled_model,self.distilled_device,self.input)

                    # Map the model's output to the corresponding decision option
                    decision = reverse_label_map.get(predicted.item(), "unknown")
                    return decision


        # Fallback to the current LLM-based review moderation flow
        print('Note: agent using llm pipeline for moderation')
        prompt_template = PromptTemplate(
            input_variables=["system_description", "task", "goal", "decision_options", "context", "review_text"],
            template="""
            System Description: {system_description}

            Task: {task}
            Goal: {goal}
            Decision options: {decision_options}
            
            Context: {context}

            Review to moderate:
            {review_text}

            Instruction: Respond with only one of the allowed decision options: {decision_options}. 
            Do not add any additional text or explanation. The response must be formatted exactly as one of the options.
            """
        )

        prompt_message = HumanMessage(
            content=prompt_template.format(
                system_description=self.system_description,
                task=self.task,
                goal=self.goal,
                decision_options=", ".join(self.decision_options),
                context=context,
                review_text=review_text
            )
        )

        # Generate the response
        self.output = self.llm([prompt_message]).content.strip()

        # Save the class variables to JSON with timestamped file name
        self.save_to_json(folder_path="./moderation_results")

        return self.output

    def save_to_json(self, folder_path):
        """
        Save class variables as a JSON file in the specified folder path.
        """
        data = {
            "system_description": self.system_description,
            "task": self.task,
            "goal": self.goal,
            "decision_options": self.decision_options,
            "input": self.input,
            "output": self.output,
            "llm_details": {
                "model_name": getattr(self.llm, "model_name", "unknown"),
                "temperature": getattr(self.llm, "temperature", "unknown")
            }
        }

        os.makedirs(folder_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"review_moderation_data_{timestamp}.json"
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Data saved to {file_path}")


def select_review(file_path,i):
    """
    Opens a YAML file and randomly selects a review from the list.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: A randomly selected review with its text and label.
    """
    # Open and load the YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Get the list of reviews
    reviews = data.get('reviews', [])

    # Ensure the list is not empty
    if not reviews:
        raise ValueError("The 'reviews' list is empty or missing in the YAML file.")

    # Randomly select and return a review
    return reviews[i]


if __name__ == "__main__":

    for i in range(100):

        print('simulating live moderation of review')

        # Initialize the ChatGPT model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

        # Initialize the agent
        agent = ReviewModerationAgent(
            llm=llm,
            system_description="A moderation system designed to flag offensive or inappropriate product reviews.",
            task="Moderate product reviews",
            goal="Determine if a review is offensive",
            decision_options=["offensive", "appropriate", "spam", "toxic"],
            allow_distilled_model=True,
            distilled_model_accuracy_threshold=60
        )

        # Example usage
        file_path = "datasets/moderation.yaml"
        random_review = select_review(file_path,i)
        print(f"Selected Review: {random_review['text']}")
        print(f"Known Review Label: {random_review['label']}")

        # Define the product review
        review_text = random_review['text']

        # Context for moderation
        context = ""

        # Moderate the review
        start_time = time.time()
        result = agent.moderate_review(review_text, context=context)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Predicted Review Label: {result}")
        print(f"Agent Time taken: {elapsed_time:.4f} seconds")

        time.sleep(0.5)

