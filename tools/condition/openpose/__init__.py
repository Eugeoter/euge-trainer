# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


from typing import Tuple, List, Callable, Union, Optional
from waifuset import logging
from .animalpose import draw_animalposes
from .types import HandResult, FaceResult, HumanPoseResult, AnimalPoseResult
from .face import Face
from .hand import Hand
from .body import Body, BodyResult, Keypoint
from . import util
import numpy as np
import torch
import os
from ..utils import MODELS_DIR, DEVICE

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


CONTROL_TYPE = "openpose"
LOGGER = logging.getLogger(CONTROL_TYPE)

DEFAULT_MODEL_REPO_ID = "lllyasviel/Annotators"
BODY_MODEL_NAME = "body_pose_model.pth"
HAND_MODEL_NAME = "hand_pose_model.pth"
FACE_MODEL_NAME = "facenet.pth"

ONNX_DET_REPO_ID = "yzd-v/DWPose"
ONNX_POSE_REPO_ID = "yzd-v/DWPose"
ANIMALPOSE_REPO_ID = "bdsqlsz/qinglong_controlnet-lllite"

ONNX_DET_NAME = "yolox_l.onnx"
ONNX_POSE_NAME = "dw-ll_ucoco_384.onnx"
ANIMALPOSE_NAME = "Annotators/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.onnx"

DEFAULT_DETECTOR = None


def draw_poses(
    poses: List[HumanPoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True
):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[HumanPoseResult]): A list of HumanPoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(
    pose_json: dict,
) -> Tuple[List[HumanPoseResult], List[AnimalPoseResult], int, int]:
    """Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.

    Returns:
        human_poses
        animal_poses
        canvas_height
        canvas_width
    """
    height = pose_json["canvas_height"]
    width = pose_json["canvas_width"]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    def decompress_keypoints(
        numbers: Optional[List[float]],
    ) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None

        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [create_keypoint(x, y, c) for x, y, c in chunks(numbers, n=3)]

    return (
        [
            HumanPoseResult(
                body=BodyResult(
                    keypoints=decompress_keypoints(pose.get("pose_keypoints_2d"))
                ),
                left_hand=decompress_keypoints(pose.get("hand_left_keypoints_2d")),
                right_hand=decompress_keypoints(pose.get("hand_right_keypoints_2d")),
                face=decompress_keypoints(pose.get("face_keypoints_2d")),
            )
            for pose in pose_json.get("people", [])
        ],
        [decompress_keypoints(pose) for pose in pose_json.get("animals", [])],
        height,
        width,
    )


def encode_poses_as_json(
    poses: List[HumanPoseResult],
    animals: List[AnimalPoseResult],
    canvas_height: int,
    canvas_width: int,
) -> dict:
    """Encode the pose as a JSON compatible dict following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """

    def compress_keypoints(
        keypoints: Union[List[Keypoint], None]
    ) -> Union[List[float], None]:
        if not keypoints:
            return None

        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        "people": [
            {
                "pose_keypoints_2d": compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d": compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        "animals": [compress_keypoints(animal) for animal in animals],
        "canvas_height": canvas_height,
        "canvas_width": canvas_width,
    }


class OpenposeDetector:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """

    def __init__(self):
        self.device = DEVICE
        self.body_estimation = None
        self.hand_estimation = None
        self.face_estimation = None

        self.dw_pose_estimation = None
        self.animal_pose_estimation = None

    def load_model(self):
        """
        Load the Openpose body, hand, and face models.
        """
        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        body_model_path = os.path.join(model_dir, "body_pose_model.pth")
        hand_model_path = os.path.join(model_dir, "hand_pose_model.pth")
        face_model_path = os.path.join(model_dir, "facenet.pth")

        if not os.path.exists(body_model_path):
            from huggingface_hub import hf_hub_download
            body_model_path = hf_hub_download(
                repo_id=DEFAULT_MODEL_REPO_ID,
                filename=BODY_MODEL_NAME,
                local_dir=model_dir,
            )

        if not os.path.exists(hand_model_path):
            from huggingface_hub import hf_hub_download
            hand_model_path = hf_hub_download(
                repo_id=DEFAULT_MODEL_REPO_ID,
                filename=HAND_MODEL_NAME,
                local_dir=model_dir,
            )

        if not os.path.exists(face_model_path):
            from huggingface_hub import hf_hub_download
            face_model_path = hf_hub_download(
                repo_id=DEFAULT_MODEL_REPO_ID,
                filename=FACE_MODEL_NAME,
                local_dir=model_dir,
            )

        self.body_estimation = Body(body_model_path)
        self.hand_estimation = Hand(hand_model_path)
        self.face_estimation = Face(face_model_path)

    def load_dw_model(self):
        from .wholebody import Wholebody  # DW Pose

        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        onnx_det_model_path = os.path.join(model_dir, ONNX_DET_NAME)
        onnx_pose_model_path = os.path.join(model_dir, ONNX_POSE_NAME)

        if not os.path.exists(onnx_det_model_path):
            from huggingface_hub import hf_hub_download
            onnx_det_model_path = hf_hub_download(
                repo_id=ONNX_DET_REPO_ID,
                filename=ONNX_DET_NAME,
                local_dir=model_dir,
            )

        if not os.path.exists(onnx_pose_model_path):
            from huggingface_hub import hf_hub_download
            onnx_pose_model_path = hf_hub_download(
                repo_id=ONNX_POSE_REPO_ID,
                filename=ONNX_POSE_NAME,
                local_dir=model_dir,
            )

        self.dw_pose_estimation = Wholebody(onnx_det_model_path, onnx_pose_model_path)

    def load_animalpose_model(self):
        from .animalpose import AnimalPose  # Animalpose

        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        onnx_det_model_path = os.path.join(model_dir, ONNX_DET_NAME)
        onnx_pose_model_path = os.path.join(model_dir, ANIMALPOSE_NAME)

        if not os.path.exists(onnx_det_model_path):
            from huggingface_hub import hf_hub_download
            onnx_det_model_path = hf_hub_download(
                repo_id=ONNX_DET_REPO_ID,
                filename=ONNX_DET_NAME,
                local_dir=model_dir,
            )

        if not os.path.exists(onnx_pose_model_path):
            from huggingface_hub import hf_hub_download
            onnx_pose_model_path = hf_hub_download(
                repo_id=ANIMALPOSE_REPO_ID,
                filename=ANIMALPOSE_NAME,
                local_dir=model_dir,
            )

        self.animal_pose_estimation = AnimalPose(onnx_det_model_path, onnx_pose_model_path)

    def unload_model(self):
        """
        Unload the Openpose models by moving them to the CPU.
        Note: DW Pose models always run on CPU, so no need to `unload` them.
        """
        if self.body_estimation is not None:
            self.body_estimation.model.to("cpu")
            self.hand_estimation.model.to("cpu")
            self.face_estimation.model.to("cpu")

    def detect_hands(
        self, body: BodyResult, oriImg
    ) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y: y + w, x: x + w, :]).astype(
                np.float32
            )
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(
                    W
                )
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(
                    H
                )

                hand_result = [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None

        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y: y + w, x: x + w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(
            np.float32
        )
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

        return None

    def detect_poses(
        self, oriImg, include_hand=False, include_face=False
    ) -> List[HumanPoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[HumanPoseResult]: A list of HumanPoseResult objects containing the detected poses.
        """
        if self.body_estimation is None:
            self.load_model()

        self.body_estimation.model.to(self.device)
        self.hand_estimation.model.to(self.device)
        self.face_estimation.model.to(self.device)

        self.body_estimation.cn_device = self.device
        self.hand_estimation.cn_device = self.device
        self.face_estimation.cn_device = self.device

        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)

                results.append(
                    HumanPoseResult(
                        BodyResult(
                            keypoints=[
                                Keypoint(
                                    x=keypoint.x / float(W), y=keypoint.y / float(H)
                                )
                                if keypoint is not None
                                else None
                                for keypoint in body.keypoints
                            ],
                            total_score=body.total_score,
                            total_parts=body.total_parts,
                        ),
                        left_hand,
                        right_hand,
                        face,
                    )
                )

            return results

    def detect_poses_dw(self, oriImg) -> List[HumanPoseResult]:
        """
        Detect poses in the given image using DW Pose:
        https://github.com/IDEA-Research/DWPose

        Args:
            oriImg (numpy.ndarray): The input image for pose detection.

        Returns:
            List[HumanPoseResult]: A list of HumanPoseResult objects containing the detected poses.
        """
        from .wholebody import Wholebody  # DW Pose

        self.load_dw_model()

        with torch.no_grad():
            keypoints_info = self.dw_pose_estimation(oriImg.copy())
            return Wholebody.format_result(keypoints_info)

    def detect_poses_animal(self, oriImg) -> List[AnimalPoseResult]:
        """
        Detect poses in the given image using RTMPose AP10k model:
        https://github.com/abehonest/ControlNet_AnimalPose

        Args:
            oriImg (numpy.ndarray): The input image for pose detection.

        Returns:
            A list of AnimalPoseResult objects containing the detected animal poses.
        """

        self.load_animalpose_model()

        with torch.no_grad():
            return self.animal_pose_estimation(oriImg.copy())

    def __call__(
        self,
        oriImg,
        include_body=True,
        include_hand=False,
        include_face=False,
        use_dw_pose=False,
        use_animal_pose=False,
        json_pose_callback: Callable[[str], None] = None,
    ):
        """
        Detect and draw poses in the given image.

        Args:
            oriImg (numpy.ndarray): The input image for pose detection and drawing.
            include_body (bool, optional): Whether to include body keypoints. Defaults to True.
            include_hand (bool, optional): Whether to include hand keypoints. Defaults to False.
            include_face (bool, optional): Whether to include face keypoints. Defaults to False.
            use_dw_pose (bool, optional): Whether to use DW pose detection algorithm. Defaults to False.
            json_pose_callback (Callable, optional): A callback that accepts the pose JSON string.

        Returns:
            numpy.ndarray: The image with detected and drawn poses.
        """
        H, W, _ = oriImg.shape
        animals = []
        poses = []
        if use_animal_pose:
            animals = self.detect_poses_animal(oriImg)
        elif use_dw_pose:
            poses = self.detect_poses_dw(oriImg)
        else:
            poses = self.detect_poses(oriImg, include_hand, include_face)

        if json_pose_callback:
            json_pose_callback(encode_poses_as_json(poses, animals, H, W))

        if poses:
            assert len(animals) == 0
            return draw_poses(
                poses,
                H,
                W,
                draw_body=include_body,
                draw_hand=include_hand,
                draw_face=include_face,
            )
        else:
            return draw_animalposes(animals, H, W)


def get_openpose(np_img: np.ndarray, include_body=True, include_hand=False, include_face=False, use_dw_pose=False, use_animal_pose=False) -> np.ndarray:
    r"""
    Get the pose of an numpy image using the Openpose model.
    """
    global DEFAULT_DETECTOR
    if DEFAULT_DETECTOR is None:
        DEFAULT_DETECTOR = OpenposeDetector()
    return DEFAULT_DETECTOR(np_img, include_body, include_hand, include_face, use_dw_pose, use_animal_pose)


def get_dwpose(np_img: np.ndarray, include_body=True, include_hand=False, include_face=False) -> np.ndarray:
    r"""
    Get the pose of an numpy image using the DWpose model.
    """
    global DEFAULT_DETECTOR
    if DEFAULT_DETECTOR is None:
        DEFAULT_DETECTOR = OpenposeDetector()
    return DEFAULT_DETECTOR(np_img, include_body, include_hand, include_face, use_dw_pose=True)


def get_animalpose(np_img: np.ndarray, include_body=True, include_hand=False, include_face=False) -> np.ndarray:
    r"""
    Get the pose of an numpy image using the Animalpose model.
    """
    global DEFAULT_DETECTOR
    if DEFAULT_DETECTOR is None:
        DEFAULT_DETECTOR = OpenposeDetector()
    return DEFAULT_DETECTOR(np_img, include_body, include_hand, include_face, use_animal_pose=True)
