import datetime
import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

DATA_LIST = {
    "object_interaction": "star/Charades_v1_480",
    "action_sequence": "star/Charades_v1_480",
    "action_prediction": "star/Charades_v1_480",
    "action_localization": "sta/sta_video",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
    "episodic_reasoning": "tvqa/frames_fps3_hq",
}

# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
# base_cache_dir = os.path.expanduser(hf_home)
base_cache_dir = '/share_1/users/bonan_ding/PAVE_data/MVBench/'

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)


cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


_MV_BENCH_VIDEO_REGEX = re.compile(
    r"^([^_]+)_(?:-?\d+(?:\.\d+)?|nan)_(?:-?\d+(?:\.\d+)?|nan)\.mp4$",
    re.IGNORECASE,
)
_MV_BENCH_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def _normalize_mvbench_video_filename(filename: str) -> str:
    """Collapse patterns like '4NI15V_4.700000000000003_25.4.mp4' to '4NI15V.mp4'."""
    if not isinstance(filename, str):
        return filename
    if not filename.endswith(".mp4"):
        return filename
    match = _MV_BENCH_VIDEO_REGEX.match(filename)
    if not match:
        return filename
    base = match.group(1)
    return f"{base}.mp4"


def _candidate_mvbench_video_filenames(filename: str) -> List[str]:
    normalized = _normalize_mvbench_video_filename(filename)
    candidates = []
    seen = set()

    def add(candidate: str):
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    add(normalized)
    base, ext = os.path.splitext(normalized)
    if ext.lower() in _MV_BENCH_VIDEO_EXTS:
        for alt_ext in _MV_BENCH_VIDEO_EXTS:
            add(base + alt_ext)
    return candidates


def _strip_trailing_numeric_suffix(value: str) -> str:
    if "_" not in value:
        return value
    parts = value.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return value


def _candidate_mvbench_frame_keys(filename: str, dataset_folder: str) -> List[str]:
    video_key = _normalize_mvbench_video_filename(filename)
    candidates = []
    seen = set()

    def add(candidate: str):
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    if "tvqa" in dataset_folder:
        base_key = video_key.split(".")[0].rstrip("_")
        add(base_key)
        add(_strip_trailing_numeric_suffix(base_key))
    else:
        add(video_key)
        add(_strip_trailing_numeric_suffix(video_key))

    return candidates


def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = base_cache_dir
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    candidate_filenames = _candidate_mvbench_video_filenames(doc["video"])
    candidate_paths = [os.path.join(cache_dir, dataset_folder, filename) for filename in candidate_filenames]
    video_path = candidate_paths[0]
    base_path, ext = os.path.splitext(video_path)
    if ext.lower() in _MV_BENCH_VIDEO_EXTS:
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return [candidate_path]

        question_id = doc.get("question_id") or doc.get("qid") or doc.get("id") or "unknown"
        sub_task = lmms_eval_specific_kwargs.get("sub_task", "unknown")
        eval_logger.warning(
            f"MISSING_VIDEO: task={sub_task}, question_id={question_id}, video={doc['video']}, "
            f"expected_paths={candidate_paths}"
        )
        return None
    return [video_path]


def mvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = base_cache_dir
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    candidate_keys = _candidate_mvbench_frame_keys(doc["video"], dataset_folder)
    candidate_paths = [os.path.join(cache_dir, dataset_folder, video_key) for video_key in candidate_keys]

    question_id = doc.get("question_id") or doc.get("qid") or doc.get("id") or "unknown"
    sub_task = lmms_eval_specific_kwargs.get("sub_task", "unknown")
    for candidate_path in candidate_paths:
        if os.path.isdir(candidate_path) and os.path.exists(candidate_path):
            return [candidate_path]

    eval_logger.warning(
        f"MISSING_VIDEO_FRAMES: task={sub_task}, question_id={question_id}, video={doc['video']}, "
        f"expected_paths={candidate_paths}"
    )
    return None


def mvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"

    full_text = "Question:" + doc["question"] + "\nOption:\n" + option_prompt + lmms_eval_specific_kwargs["post_prompt"]
    return full_text


def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def mvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mvbench_perception_score), value: metric value
    """
    pred = results[0]

    # Handle missing video cases
    if pred == "Unable to process: video file not found" or pred == "Error in loading video":
        question_id = doc.get("question_id") or doc.get("qid") or doc.get("id") or "unknown"
        eval_logger.info(f"SKIPPED_EVALUATION: question_id={question_id}, video={doc.get('video', 'unknown')}, reason=missing_video")
        # Set empty prediction so it gets excluded from accuracy calculation
        pred = ""

    # Calculate the ground truth option letter
    option_letters = string.ascii_uppercase
    gt_option_letter = None
    for i, candidate in enumerate(doc["candidates"]):
        if candidate == doc["answer"]:
            gt_option_letter = option_letters[i]
            break

    # Calculate the score using mcq_acc function
    score = mcq_acc(gt_option_letter, pred)

    data_dict = {"pred_answer": pred, "gt_answer": gt_option_letter, "score": score}

    return {"mvbench_accuracy": data_dict}


def mvbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_answered = 0
    total_correct = 0
    total_skipped = 0
    
    for result in results:
        if result["pred_answer"] != "":
            total_answered += 1
            total_correct += result["score"]
        else:
            total_skipped += 1

    total_questions = len(results)
    accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    
    eval_logger.info(f"MVBench Evaluation Summary: {total_questions} total questions, {total_answered} answered, {total_skipped} skipped (missing videos), accuracy: {accuracy:.2f}%")
    
    return accuracy
