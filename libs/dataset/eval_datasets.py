import os
import json
from typing import Any, Dict, List, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from libs.utils.video_io import SafeVideoReader
import decord

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_video_frames_uniform(video_path: str, num_frames: int) -> np.ndarray:
    vr = SafeVideoReader(video_path)
    total_frame_num = len(vr)
    if total_frame_num <= 0:
        raise ValueError(f"No frames in video: {video_path}")
    import numpy as _np

    indices = _np.linspace(0, total_frame_num - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices)
    if hasattr(decord, "ndarray") and isinstance(frames, decord.ndarray.NDArray):
        frames = frames.asnumpy()
    elif hasattr(frames, "numpy"):
        frames = frames.numpy()
    return frames


class AVQADataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        video_folder: str,
        feat_folder: str,
        image_processor: Any,
        for_get_frames_num: int,
    ) -> None:
        annotations = json.load(open(annotation_file))
        self.start_prompt = (
            "Select the best answer to the following multiple-choice question "
            "based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n"
        )
        self.end_prompt = "Answer with the option's letter from the given choices directly."
        self.mapping = {0: "A", 1: "B", 2: "C", 3: "D"}

        all_items: List[Dict[str, str]] = []
        for ele in annotations:
            vid = ele["video_name"]
            curr_video_name = vid + ".mp4"
            curr_feat_name = vid + ".pt"

            question = self.start_prompt + ele["question_text"] + "\n"
            options = ele["multi_choice"]
            refined_options = (
                "A." + options[0] + "\n"
                + "B." + options[1] + "\n"
                + "C." + options[2] + "\n"
                + "D." + options[3] + "\n"
            )
            question = question + refined_options + self.end_prompt
            answer = self.mapping[ele["answer"]]

            video_path = os.path.join(video_folder, curr_video_name)
            feat_path = os.path.join(feat_folder, curr_feat_name)
            if os.path.exists(video_path) and os.path.exists(feat_path):
                all_items.append(
                    {
                        "vid": vid,
                        "video_path": video_path,
                        "feat_path": feat_path,
                        "question": question,
                        "answer": answer,
                    }
                )

        self.items = all_items
        self.image_processor = image_processor
        self.for_get_frames_num = for_get_frames_num

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        if not os.path.exists(item["video_path"]):
            raise FileNotFoundError(f"Video not found: {item['video_path']}")
        frames = load_video_frames_uniform(item["video_path"], self.for_get_frames_num)
        video = self.image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"
        ]
        video = [video]

        audio_feature = (
            torch.load(item["feat_path"], map_location="cpu")
            if os.path.exists(item["feat_path"])
            else torch.zeros(1, 10, 1024)
        )
        if getattr(audio_feature, "requires_grad", False):
            audio_feature.requires_grad = False

        return {
            "vid": item["vid"],
            "video": video,
            "audio_feature": audio_feature,
            "question": item["question"],
            "answer": item["answer"],
        }


class AVSDDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        video_folder: str,
        feat_folder: str,
        image_processor: Any,
        for_get_frames_num: int,
    ) -> None:
        annotations = json.load(open(annotation_file))
        dialogs = annotations["dialogs"]
        all_items: List[Dict[str, Any]] = []
        for ele in dialogs:
            vid = ele["image_id"]
            video_path = os.path.join(video_folder, vid + ".mp4")
            feat_path = os.path.join(feat_folder, vid + ".pt")
            if os.path.exists(video_path) and os.path.exists(feat_path):
                all_items.append(
                    {
                        "vid": vid,
                        "video_path": video_path,
                        "feat_path": feat_path,
                        "conversation": ele["dialog"],
                    }
                )
        self.items = all_items
        self.image_processor = image_processor
        self.for_get_frames_num = for_get_frames_num

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        frames = load_video_frames_uniform(item["video_path"], self.for_get_frames_num)
        video = self.image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"
        ]
        video = [video]

        audio_feature = (
            torch.load(item["feat_path"], map_location="cpu")
            if os.path.exists(item["feat_path"])
            else torch.zeros(1, 10, 1024)
        )
        if getattr(audio_feature, "requires_grad", False):
            audio_feature.requires_grad = False

        return {
            "vid": item["vid"],
            "video": video,
            "audio_feature": audio_feature,
            "conversation": item["conversation"],
        }


class MusicAVQADataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        video_folder: str,
        feature_folder: str,
        image_processor: Any,
        feat_type: Literal["languagebind", "imagebind", "audio", "other"],
        for_get_frames_num: int,
    ) -> None:
        annotations = json.load(open(annotation_file))
        self.image_processor = image_processor
        self.for_get_frames_num = for_get_frames_num
        self.feat_type = feat_type
        items: List[Dict[str, str]] = []
        for ele in annotations:
            vid = ele["video_id"]
            curr_video_name = vid + ".mp4"
            curr_feat_name = vid + ".pt"
            video_subfolder = (
                "MUSIC-AVQA-videos-Real" if vid[0].isdigit() else "MUCIS-AVQA-videos-Synthetic"
            )
            video_path = os.path.join(video_folder, video_subfolder, curr_video_name)
            feat_subfolder = "MUSIC-AVQA-videos-Real_audio_imagebind_feat" if vid[0].isdigit() else "MUCIS-AVQA-videos-Synthetic_audio_imagebind_feat"
            feat_path = os.path.join(feature_folder, feat_subfolder, curr_feat_name)
            items.append(
                {
                    "vid": vid,
                    "video_path": video_path,
                    "feat_path": feat_path,
                    "question": ele["question_content"],
                    "answer": ele["anser"],
                }
            )
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        if not os.path.exists(item["video_path"]):
            raise FileNotFoundError(f"Video not found: {item['video_path']}")
        frames = load_video_frames_uniform(item["video_path"], self.for_get_frames_num)
        video = self.image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"
        ]
        video = [video]
        audio_feature = (
            torch.load(item["feat_path"], map_location="cpu")
            if os.path.exists(item["feat_path"])
            else torch.zeros(1, 10, 1024)
        )
        if getattr(audio_feature, "requires_grad", False):
            audio_feature.requires_grad = False
        return {
            "vid": item["vid"],
            "video": video,
            "audio_feature": audio_feature,
            "question": item["question"],
            "answer": item["answer"],
        }


class ScanQADataset(Dataset):
    def __init__(
        self,
        question_file: str,
        video_folder: str,
        feature_folder: str,
        image_processor: Any,
    ) -> None:
        question_file_content = json.load(open(question_file))
        all_items: List[Dict[str, Any]] = []
        for curr_q in question_file_content:
            scene_dir = os.path.join(video_folder, curr_q["scene_id"])
            feat_pt = os.path.join(
                feature_folder, curr_q["scene_id"], "video_features.pt"
            )
            if os.path.exists(scene_dir) and os.path.exists(feat_pt):
                all_items.append(
                    {
                        "question_id": curr_q["question_id"],
                        "scene_folder": scene_dir,
                        "scane_feature_path": feat_pt,
                        "question": curr_q["text"],
                        "answer": curr_q["answers"],
                    }
                )
        self.items = all_items
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import numpy as _np
        from PIL import Image as _Image

        item = self.items[idx]
        scene_folder = item["scene_folder"]
        npy_path = os.path.join(scene_folder, "stacked_images.npy")
        if os.path.exists(npy_path):
            images_array = np.load(npy_path)
        else:
            all_files = os.listdir(scene_folder)
            jpg_files = sorted(
                [f for f in all_files if f.endswith(".jpg")],
                key=lambda x: int(os.path.splitext(x)[0]),
            )
            images: List[np.ndarray] = []
            for fn in jpg_files:
                img = _Image.open(os.path.join(scene_folder, fn)).convert("RGB")
                images.append(_np.array(img))
            if len(images) > 32:
                images = images[:32]
            elif 0 < len(images) < 32:
                images = images + [images[-1]] * (32 - len(images))
            if not images:
                raise ValueError(f"No JPG images found in {scene_folder}")
            images_array = _np.stack(images, axis=0)
            np.save(npy_path, images_array)

        video = self.image_processor.preprocess(images_array, return_tensors="pt")[
            "pixel_values"
        ]
        video = [video]

        video_feat = torch.load(item["scane_feature_path"], map_location="cpu")
        bsz, _, dim = video_feat.shape
        v, h, w = 32, 24, 24
        video_feat = video_feat.view(bsz, v, h, w, dim).squeeze(0).permute(3, 0, 1, 2)

        return {
            "question_id": item["question_id"],
            "video": video,
            "video_feat": video_feat,
            "question": item["question"],
            "answer": item["answer"],
        }


class SQA3DDataset(Dataset):
    """SQA3D Dataset - 3D Scene Question Answering"""
    def __init__(
        self,
        question_file: str,
        video_folder: str,
        feature_folder: str,
        image_processor: Any,
    ) -> None:
        question_file_content = json.load(open(question_file))
        all_items: List[Dict[str, Any]] = []
        for curr_q in question_file_content:
            scene_dir = os.path.join(video_folder, curr_q["scene_id"])
            feat_pt = os.path.join(
                feature_folder, curr_q["scene_id"], "video_features.pt"
            )
            if os.path.exists(scene_dir) and os.path.exists(feat_pt):
                all_items.append(
                    {
                        "question_id": curr_q["question_id"],
                        "scene_folder": scene_dir,
                        "scane_feature_path": feat_pt,
                        "question": curr_q["text"],
                        "answer": curr_q["answers"],
                    }
                )
        print(f"SQA3D: Loaded {len(all_items)} items from {len(question_file_content)} total questions")
        self.items = all_items
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import numpy as _np
        from PIL import Image as _Image

        item = self.items[idx]
        scene_folder = item["scene_folder"]
        npy_path = os.path.join(scene_folder, "stacked_images.npy")
        if os.path.exists(npy_path):
            images_array = np.load(npy_path)
        else:
            all_files = os.listdir(scene_folder)
            jpg_files = sorted(
                [f for f in all_files if f.endswith(".jpg")],
                key=lambda x: int(os.path.splitext(x)[0]),
            )
            images: List[np.ndarray] = []
            for fn in jpg_files:
                img = _Image.open(os.path.join(scene_folder, fn)).convert("RGB")
                images.append(_np.array(img))
            if len(images) > 32:
                images = images[:32]
            elif 0 < len(images) < 32:
                images = images + [images[-1]] * (32 - len(images))
            if not images:
                raise ValueError(f"No JPG images found in {scene_folder}")
            images_array = _np.stack(images, axis=0)
            np.save(npy_path, images_array)

        video = self.image_processor.preprocess(images_array, return_tensors="pt")[
            "pixel_values"
        ]
        video = [video]

        video_feat = torch.load(item["scane_feature_path"], map_location="cpu")
        bsz, _, dim = video_feat.shape
        v, h, w = 32, 24, 24
        video_feat = video_feat.view(bsz, v, h, w, dim).squeeze(0).permute(3, 0, 1, 2)

        return {
            "question_id": item["question_id"],
            "video": video,
            "video_feat": video_feat,
            "question": item["question"],
            "answer": item["answer"],
        }
