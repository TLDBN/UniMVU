import logging
import threading
from typing import List

import cv2
import decord
import numpy as np
import torch


logger = logging.getLogger(__name__)


class SafeVideoReader(decord.VideoReader):
    """
    Safe VideoReader that prevents deadlocks with threading and OpenCV fallback.
    Key improvements:
    1. Threading with timeout prevents deadlocks
    2. Automatic fallback to OpenCV when decord fails/hangs
    3. DataLoader-safe (no nested multiprocessing)
    4. Robust error handling and logging
    """

    def __init__(self, uri, ctx=decord.cpu(0), width: int = -1, height: int = -1, num_threads: int = 0, fault_tol: int = -1):
        super().__init__(uri, ctx, width, height, num_threads, fault_tol)
        self.uri = uri
        self._width = width if width > 0 else None
        self._height = height if height > 0 else None

    def get_batch(self, indices: List[int]):
        """Get batch of frames with timeout protection and OpenCV fallback."""
        result = []
        exception = []

        def target():
            decord.bridge.set_bridge('torch')
            try:
                res = super(SafeVideoReader, self).get_batch(indices)
                result.append(res)
            except Exception as e:
                exception.append(e)

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=100)

        if thread.is_alive():
            logger.warning(f"Timeout reading {self.uri} with decord, switching to OpenCV")
            return self._get_batch_opencv(indices)
        elif exception:
            logger.warning(f"Error using decord: {exception[0]}, switching to OpenCV")
            return self._get_batch_opencv(indices)
        else:
            return result[0]

    def _get_batch_opencv(self, indices: List[int]):
        """Fallback method using OpenCV for reliable frame reading."""
        cap = cv2.VideoCapture(self.uri)
        frames = []
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine final resolution
        target_width = self._width if self._width is not None else original_width
        target_height = self._height if self._height is not None else original_height

        for idx in sorted(indices):  # Sort indices to improve reading efficiency
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                # Generate a placeholder black frame
                frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            else:
                # Convert color space and resize if necessary
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (target_width, target_height) != (original_width, original_height):
                    frame = cv2.resize(frame, (target_width, target_height))

            frames.append(frame)

        cap.release()

        # Convert to a torch tensor while maintaining consistent dimensions
        frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])
        return frames_tensor
