# Breast Cancer Detection with Azure Custom Vision

This project uses Azure Custom Vision to perform object detection on breast cancer images. It predicts and visualizes bounding boxes for detected regions in input images.

## Features

- Loads a test image and sends it to a deployed Azure Custom Vision model.
- Draws bounding boxes and labels for detected objects with probability above 50%.
- Saves and displays the annotated image.

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   Create a `.env` file in the project root with the following content:
   ```
   PREDICTION_KEY=your_azure_prediction_key
   ENDPOINT=your_azure_endpoint
   PROJECT_ID=your_custom_vision_project_id
   PUBLISHED_NAME=your_published_model_name
   ```

4. **Prepare test images**:

   Place your test images in the `data/test/` directory.

## Usage

Run the main script:
```
python main.py
```

The script will:
- Load the specified image,
- Run object detection,
- Draw bounding boxes and labels,
- Save the result to `data/predicted/object_detection_result.jpg`,
- Display the annotated image.

## Requirements

- Python 3.7+
- Azure Cognitive Services Custom Vision SDK
- Pillow
- python-dotenv

Install all requirements using:
```
pip install -r requirements.txt
```

## Notes

- Make sure your Azure Custom Vision model is published and accessible.
- Update `IMAGE_PATH` in `main.py` to test different images.

## License

This project is for educational purposes.
