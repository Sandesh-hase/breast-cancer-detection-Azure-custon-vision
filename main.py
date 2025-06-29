from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure config
PREDICTION_KEY = os.getenv("PREDICTION_KEY")
ENDPOINT = os.getenv("ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
PUBLISHED_NAME = os.getenv("PUBLISHED_NAME")
IMAGE_PATH = "data/test/malignant (29).png"

# Authenticate prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
prediction_client = CustomVisionPredictionClient(endpoint=ENDPOINT, credentials=credentials)

# Load image as binary
with open(IMAGE_PATH, "rb") as image_file:
    image_data = image_file.read()

# Perform object detection
results = prediction_client.detect_image(PROJECT_ID, PUBLISHED_NAME, image_data)

# Open image for drawing
image = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(image)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Draw bounding boxes and labels
for prediction in results.predictions:
    if prediction.probability > 0.5:
        tag = prediction.tag_name
        prob = prediction.probability
        bbox = prediction.bounding_box

        # Convert normalized coordinates to actual pixel values
        left = bbox.left * image.width
        top = bbox.top * image.height
        width = bbox.width * image.width
        height = bbox.height * image.height
        right = left + width
        bottom = top + height

        # Draw bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # Create label
        label = f"{tag} ({prob:.0%})"

        # Get text size using getbbox
        bbox_text = font.getbbox(label)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Calculate label background position
        label_bg_top = max(0, top - text_height - 4)

        # Draw background rectangle for label
        draw.rectangle(
            [left, label_bg_top, left + text_width + 4, label_bg_top + text_height + 4],
            fill="red"
        )

        # Draw label text
        draw.text((left + 2, label_bg_top + 2), label, fill="white", font=font)

# Save and show image
output_path = "data/predicted/object_detection_result.jpg"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
image.save(output_path)

print(f"Detection results saved to {output_path}")
image.show()
