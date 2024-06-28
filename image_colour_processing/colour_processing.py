from typing import Tuple, Callable, List
from mask_step import MaskStep

import cv2, cv2_helper
from numpy import ndarray
import time
import numpy as np

class ColourProcessing:
    def __init__(self, image_scaling: float, display_scaling: float, mask_steps: Tuple[MaskStep]):
        """
        """
        self._in_tuning_mode: bool
        self._mask_steps: Tuple[MaskStep] = mask_steps
        self._image_scaling: float = image_scaling
        self._display_scaling: float = display_scaling
    
        if len(mask_steps) == 0:
            raise IndexError("Must give at least one MaskStep to ColourProcessing")
        
        for step in mask_steps:
            step.set_display_scaling(display_scaling)
        
    def process_mask(self, image: ndarray) -> ndarray:
        """
        Take an image and create a mask by running it through all the MaskSteps of this object.

        Paramters:
            image (ndarray): The image to process

        Returns:
            ndarray: The mask for the image
        """
        start = time.time()
        mask_timings = []
        processed_image = image
        for mask_step in self._mask_steps:
            step_start = time.time()
            processed_image = mask_step.process(image, processed_image)
            mask_timings.append(time.time() - step_start)

        print(f"Process time: {time.time() - start:.4f}. Steps: {np.around(mask_timings, 4)}")
        return processed_image
    
    def single_image_mask(self, filename: str):
        image: ndarray = cv2.imread(filename)
        print(f"Image size: {image.shape}")
        image = cv2.resize(image, None, fx=self._image_scaling, fy=self._image_scaling, interpolation = cv2.INTER_AREA)
        print(f"Image size scaled: {image.shape}")

        for i in range(0, 10):
            self.process_mask(image)

    def mask_step_tuning(self, image: ndarray, starting_step: int = 0) -> bool:
        image = cv2.resize(image, None, fx=self._image_scaling, fy=self._image_scaling, interpolation = cv2.INTER_AREA)

        mask_step_index: int = starting_step
        self._mask_steps[mask_step_index].start_display()
        
        while True:
            self.process_mask(image)            
            
            key_pressed = cv2.waitKey(5)
            new_index = None

            if key_pressed == cv2_helper.WaitKeyStroke.ESC or key_pressed == ord('s') or key_pressed == ord('d'):
                self._mask_steps[mask_step_index].stop_display()
                return key_pressed, mask_step_index
            
            elif key_pressed == ord('w'):
                new_index = max(mask_step_index - 1, 0)
            elif key_pressed == ord('e'):
                new_index = min(mask_step_index + 1, len(self._mask_steps) - 1)

            else:
                for i in range(1, len(self._mask_steps)+1):
                    if key_pressed == ord(str(i)):
                        new_index = i - 1
                        break
                        
            if new_index is not None and new_index != mask_step_index:
                print(f"Stopping {mask_step_index} and starting {new_index}")
                self._mask_steps[mask_step_index].stop_display()
                self._mask_steps[new_index].start_display()
                mask_step_index = new_index

    def single_image_mask_tuning(self, filename: str):
        image: ndarray = cv2.imread(filename)
        print(f"Image size: {image.shape}")
        
        mask_step_index: int = 0
        self._mask_steps[mask_step_index].start_display()
        
        self.mask_step_tuning(image)
        cv2.destroyAllWindows()

        print(f"The final Mask Steps are below:\n(")
        for step in self._mask_steps:
            print(f"{repr(step)}, ")
        print(")")

    def multi_image_mask_tuning(self, image_callback: Callable[..., ndarray]):
        images: List[ndarray] = [image_callback()]
        image_index: int = 0
        mask_step: int = 0
        while True:
            key_pressed, mask_step = self.mask_step_tuning(images[image_index], mask_step)

            if key_pressed == cv2_helper.WaitKeyStroke.ESC:
                break
            elif key_pressed == ord('s'):
                image_index = max(0, image_index - 1)
            elif key_pressed == ord('d'):
                image_index += 1
                if image_index >= len(images):
                    images.append(image_callback())


        cv2.destroyAllWindows()
