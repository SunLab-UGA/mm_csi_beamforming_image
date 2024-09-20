# this looks in a file path for specific images with a suffix and postprocesses to a new directory


# crops and rotates the images to the correct orientation
# the spatial images are rotated 90deg counter clockwise
# the camera image is cropped to a left-justified square

import os
from typing import List, Tuple
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # In case of broken images


def parse_images(directory: str, suffix: str, extension: str) -> List[str]:
    """
    Crawl the given directory and return a list of image paths that match the suffix and extension.

    :param directory: Path to the directory to search.
    :param suffix: The suffix to match (e.g., 'image_suffix').
    :param extension: The image extension to match (e.g., '.png').
    :return: List of paths to the images that match.
    """
    matched_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix + extension):
                matched_images.append(os.path.join(root, file))
    return matched_images


if __name__ == "__main__":
    input_directory = '/home/sunlab/beamscan_data/trial_2'
    output_directory = '/home/sunlab/beamscan_data/trial_2_processed'
    camera_suffix = 'camera'
    camera_extension = '.jpg'
    # set to crop the 1080x1920 image to a 1080x1080 square (left-justified)
    camera_crop = (0, 0, 1080, 1080) # (left, upper, right, lower)

    spatial_suffix = 'heatmap'
    spatial_extension = '.png'
    spatial_rotate =  90 # degrees counter clockwise

    # check if the input directory exists
    if not os.path.exists(input_directory):
        print(f"Directory {input_directory} does not exist.")
        exit(1)

    # Parse images in the input directory
    cameral_images = parse_images(input_directory, camera_suffix, camera_extension)
    spatial_images = parse_images(input_directory, spatial_suffix, spatial_extension)

    print(f"Found {len(cameral_images)} camera images and {len(spatial_images)} spatial images in {input_directory}")
    if len(cameral_images) == 0 or len(spatial_images) == 0:
        print("No images found, exiting.")
        exit(1)

    # Ensure output directory exists
    os.makedirs(output_directory) # raises an error if the output directory already exists
    
    print("Processing camera images...")
    # Process camera images (crop to square, left-justified)
    for image_path in cameral_images:
        try:
            with Image.open(image_path) as img:
                img:Image.Image = img.crop(camera_crop)
                # Save processed image
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_directory, filename)
                img.save(output_path)
                print(f"Processed image {image_path} and saved to {output_path}")
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")

    print()
    print("Processing camera images complete.")
    print()
    print("Processing spatial images...")

    # Process spatial images (rotate 90 degrees counter clockwise)
    for image_path in spatial_images:
        try:
            with Image.open(image_path) as img:
                img:Image.Image = img.rotate(spatial_rotate)
                # Save processed image
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_directory, filename)
                img.save(output_path)
                print(f"Processed image {image_path} and saved to {output_path}")
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")
    
    print("Processing complete.")
