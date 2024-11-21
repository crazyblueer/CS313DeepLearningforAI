import os
import cv2
from anomalib import TaskType
from anomalib.deploy import OpenVINOInferencer
from loguru import logger

def inference():
    # Define paths
    model_path = '/home/group2/anomalib/testModel/CS313DeepLearningforAI/content/anomaly_detection/anomalib_weight/weights/openvino/model.bin'  # Path to OpenVINO model
    metadata_path = '/home/group2/anomalib/testModel/CS313DeepLearningforAI/content/anomaly_detection/anomalib_weight/weights/openvino/metadata.json'  # Path to metadata
    test_images_dir = '/home/group2/anomalib/testModel/bottle/test/broken_large'  # Path to test images
    output_dir = '/home/group2/anomalib/testModel/bottle/test/output'  # Folder where you want to save the output images

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the inferencer
    inferencer = OpenVINOInferencer(
        path=model_path,
        metadata=metadata_path,
        device="CPU",
        task=TaskType.CLASSIFICATION
    )
    logger.info('Model loaded successfully.')

    # Iterate over test images and make predictions
    for img_name in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_name)
        if os.path.isfile(img_path):  # Ensure it's a file
            # Predict the anomaly
            predictions = inferencer.predict(img_path)

            # Log the prediction
            logger.info(f"{img_name}: {predictions.pred_label}, {predictions.pred_score:.4f}")

            # Get the segmented image with the highlighted anomaly regions
            segmented_img = predictions.segmentations  # This should be the output image with the detected region

            # Save the segmented image to the output directory
            output_img_path = os.path.join(output_dir, f"segmented_{img_name}")
            cv2.imwrite(output_img_path, segmented_img)  # Save the image

if __name__ == "__main__":
    inference()
