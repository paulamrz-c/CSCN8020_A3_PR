import numpy as np
import cv2
from collections import deque
import random

class Preprocessor:
    """Prepare Atari frames."""
    def __init__(self, image_shape=(84, 80)):
        self.image_shape = image_shape

    def img_crop(self, img):
        """Cut top and bottom."""
        return img[30:-12, :, :]

    def downsample(self, img):
        """Resize to 84x80."""
        return cv2.resize(img, (80, 84))

    def transform_reward(self, reward):
        """Convert reward to -1, 0, 1."""
        return np.sign(reward)

    def to_grayscale(self, img):
        """Convert RGB to gray."""
        return np.mean(img, axis=2).astype(np.uint8)

    def normalize_grayscale(self, img):
        """Scale to [-1, 1]."""
        return (img - 128.0) / 128.0

    def process_frame(self, img):
        """Crop, resize, grayscale, normalize."""
        img = self.img_crop(img)
        img = self.downsample(img)
        img = self.to_grayscale(img)
        img = self.normalize_grayscale(img)
        return np.expand_dims(img.reshape(self.image_shape[0], self.image_shape[1], 1), axis=0)


class ReplayMemory:
    """Save and sample experiences."""
    def __init__(self, max_len=10000):
        self.max_len = max_len
        self.states = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.next_states = deque(maxlen=max_len)
        self.done_flags = deque(maxlen=max_len)

    def add(self, state, action, reward, next_state, done):
        """Add one experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done_flags.append(done)

    def sample(self, batch_size):
        """Get random batch."""
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        batch = (
            np.array([self.states[i] for i in indices]).squeeze(),
            np.array([self.actions[i] for i in indices]),
            np.array([self.rewards[i] for i in indices]),
            np.array([self.next_states[i] for i in indices]).squeeze(),
            np.array([self.done_flags[i] for i in indices])
        )
        return batch


class FrameStacker:
    """Keep last N frames (default 4)."""
    def __init__(self, num_frames=4):
        self.frames = deque(maxlen=num_frames)
        self.num_frames = num_frames
        self.prep = Preprocessor()

    def reset(self, frame):
        """Fill stack with first frame."""
        processed = self.prep.process_frame(frame)
        for _ in range(self.num_frames):
            self.frames.append(processed)
        return np.concatenate(list(self.frames), axis=-1)

    def update(self, frame):
        """Add new frame."""
        processed = self.prep.process_frame(frame)
        self.frames.append(processed)
        return np.concatenate(list(self.frames), axis=-1)

    def get_current_stack(self):
        """Return 4 stacked frames."""
        while len(self.frames) < self.num_frames:
            self.frames.append(self.frames[-1])
        return np.concatenate(list(self.frames), axis=-1)
