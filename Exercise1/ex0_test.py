import cv2
import os
import numpy as np
import unittest
from unittest import TestCase
from ex0 import IMAGE_PATH, ImageProcessor

IMAGE_PARENT_DIRECTORY: str = os.path.dirname(IMAGE_PATH)


def create_image(image_name: str) -> tuple[np.ndarray, str]:
    # Create a random image, create the full path to store it and then save it using CV2.
    test_image: np.ndarray = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    full_image_path: str = os.path.join(IMAGE_PARENT_DIRECTORY, image_name)
    test_image_bgr: np.ndarray = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_image_path, test_image_bgr)
    return test_image, full_image_path


def delete_image(full_image_path: str):
    # Remove the randomly created image.
    if os.path.exists(full_image_path):
        os.remove(full_image_path)


class Tests(TestCase):
    def test_init_rgb(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        image_rgb, image_rgb_colour_type = processor.get_image_data()

        self.assertTrue(np.array_equal(image_rgb, test_image), "RGB images are loaded incorrectly!")
        self.assertEqual(image_rgb_colour_type, "RGB", "The colour scheme is incorrect for RGB images!")

        delete_image(image_path)

    def test_init_bgr(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        image_bgr, image_rgb_colour_type = processor.get_image_data()

        self.assertTrue(np.array_equal(image_bgr[:, :, 0], test_image[:, :, 2]), "BGR images are loaded incorrectly!")
        self.assertTrue(np.array_equal(image_bgr[:, :, 1], test_image[:, :, 1]), "BGR images are loaded incorrectly!")
        self.assertTrue(np.array_equal(image_bgr[:, :, 2], test_image[:, :, 0]), "BGR images are loaded incorrectly!")
        self.assertEqual(image_rgb_colour_type, "BGR", "The colour scheme is incorrect for BGR images!")

        delete_image(image_path)

    def test_init_gray(self):
        test_image, image_path = create_image("test.png")
        test_image_shape: np.ndarray = np.array(test_image.shape)[:2]

        processor: ImageProcessor = ImageProcessor(image_path, "Gray")
        image_gray, image_rgb_colour_type = processor.get_image_data()
        image_gray_shape: np.ndarray = np.array(image_gray.shape)

        self.assertTrue(np.array_equal(image_gray_shape, test_image_shape), "Grayscale images are loaded incorrectly!")
        self.assertEqual(image_rgb_colour_type, "Gray", "The colour scheme is incorrect for grayscale images!")

        delete_image(image_path)

    def test_bgr_to_rgb(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        processor.convert_colour()
        image_converted, image_converted_colour_type = processor.get_image_data()

        self.assertTrue(np.array_equal(image_converted, test_image),
                        "The colour conversion from BGR to RGB is incorrect!")
        self.assertEqual(image_converted_colour_type, "RGB", "The colour scheme is set incorrectly")

        delete_image(image_path)

    def test_rgb_to_bgr(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        processor.convert_colour()
        image_converted, image_converted_colour_type = processor.get_image_data()

        test_image_converted: np.ndarray = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

        self.assertTrue(np.array_equal(image_converted, test_image_converted),
                        "The colour conversion from RGB to BGR is incorrect!")
        self.assertEqual(image_converted_colour_type, "BGR", "The colour scheme is set incorrectly")

        delete_image(image_path)

    def test_clip(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        processor.clip_image(60, 180)
        clipped_image, _ = processor.get_image_data()

        test_image_clipped: np.ndarray = np.clip(test_image, 60, 180)

        self.assertTrue(np.array_equal(clipped_image, test_image_clipped),
                        "The clip function does not show the correct behaviour!")

        delete_image(image_path)

    def test_flip_vertical(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        processor.flip_image(0)
        flipped_image, _ = processor.get_image_data()

        test_image_flipped: np.ndarray = np.flipud(test_image)

        self.assertTrue(np.array_equal(flipped_image, test_image_flipped),
                        "The vertical flip does not work correctly!")

        delete_image(image_path)

    def test_flip_horizontal(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        processor.flip_image(1)
        flipped_image, _ = processor.get_image_data()

        test_image_flipped: np.ndarray = np.fliplr(test_image)

        self.assertTrue(np.array_equal(flipped_image, test_image_flipped),
                        "The horizontal flip does not work correctly!")

        delete_image(image_path)

    def test_flip_both(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        processor.flip_image(2)
        flipped_image, _ = processor.get_image_data()

        test_image_flipped: np.ndarray = np.flip(test_image, axis=[0, 1])

        self.assertTrue(np.array_equal(flipped_image, test_image_flipped),
                        "Flipping the image in both directions does not work correctly!")

        delete_image(image_path)


if __name__ == '__main__':
    unittest.main()
