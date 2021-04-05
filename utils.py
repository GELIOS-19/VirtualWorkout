import cv2
import torch
import numpy as np
import posenet
import time

tracker = None

class WebcamImage():
    """
    Class for handling the extraction of images from the webcam
    """

    def __init__(self, camera_id=0, width=1280, height=720):
        
        self.video = cv2.VideoCapture(camera_id)
        self.video.set(3, width)
        self.video.set(4, height)

    def get_image(self):
        """
        Pulls the video frame from the camera and returns it
        """
        _, frame  = self.video.read()
        return frame

    def close(self):
        """
        Closes the webcam to avoid memory leakage
        """
        self.video.release()

def load_model(id=101, device="cuda"):
    """
    Load in the model.
    Uses a pretrained MobileNetv2 to do real time pose estimation
    """
    model = posenet.load_model(101)
    return model.to(device)

def get_pose_coords(image, model):
    """
    Returns the pose coords
    """
    input_image, display_image, output_scale = posenet.utils._process_input(image, 0.7125, model.output_stride)
    with torch.no_grad():
        input_image = torch.Tensor(input_image).cuda()
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=model.output_stride,
            max_pose_detections=10,
            min_pose_score=0.15)
    return keypoint_coords * output_scale, display_image, pose_scores, keypoint_scores

def get_image_and_coords(image, model):
    """
    Utility function that produces both the coords and the image with the lines
    """
    keypoint_coords, display_image, pose_scores, keypoint_scores = get_pose_coords(image, model)
    overlay_image = posenet.draw_skel_and_kp(
        display_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.15, min_part_score=0.1)
    return keypoint_coords, overlay_image


class BodyPoseEmbedding():
    """
    Class for generating pose embeddings to make pose classification easier
    """

    def __init__(self):
        self.landmarks = posenet.constants.PART_NAMES
        self.landmark_ids = posenet.constants.PART_IDS
        self.distance_pairs = [("leftShoulder", "leftElbow"), ("rightShoulder", "rightElbow"),
                                 ("leftElbow", "leftWrist"),  ("rightElbow", "rightWrist"),
                                 ("leftHip", "leftKnee"), ("rightHip", "rightKnee"),
                                 ("leftKnee", "leftAnkle"), ("rightKnee", "rightAnkle"),
                                 
                                 ("leftShoulder", "leftWrist"), ("rightShoulder", "rightWrist"),
                                 ("leftHip", "leftAnkle"), ("rightHip", "rightAnkle"),
                                 
                                 ("leftHip", "leftWrist"), ("rightHip", "rightWrist"),
                                 
                                 ("leftShoulder", "leftAnkle"), ("rightShoulder", "rightAnkle"),
                                 ("leftHip", "leftWrist"), ("rightHip", "rightWrist"),
                                 
                                 ("leftElbow", "rightElbow"), ("leftKnee", "rightKnee"),
                                 ("leftWrist", "rightWrist"), ("leftAnkle", "rightAnkle")
                                 ]

        
    
    def _get_center_and_size(self, pose, tsm=2.5):
        """
        Calculates the center of the body
        """
        # Calculate the centers of the hips and shoulders
        left_hip = pose[self.landmark_ids["leftHip"], :]
        right_hip = pose[self.landmark_ids["rightHip"], :]
        c1 = (left_hip + right_hip) / 2
        left_shoulder = pose[self.landmark_ids["leftShoulder"], :]
        right_shoulder = pose[self.landmark_ids["rightShoulder"], :]
        c2 = (left_shoulder + right_shoulder) / 2
        # Calculate sizes
        torso = np.linalg.norm(c2 - c1)
        max_dist = np.max(np.linalg.norm(pose - c1, axis=1)) # Max dist from hips center

        return (c1, c2), max(torso*tsm, max_dist)

    def get_embeddings(self, pose, tsm=2.5):
        """
        Generates the embeddings for the pose
        """
        # Normalize
        (center, shoulder_center), size = self._get_center_and_size(pose)
        pose = (pose - center) / size * 100
        # Embedding
        embedding = [center - shoulder_center]
        for part1, part2 in self.distance_pairs:
            embedding.append(pose[self.landmark_ids[part1], :] - pose[self.landmark_ids[part2], :])
        return np.concatenate(embedding)


class ExerciseTracker():
    """
    Utility class for keeping track of exercise 
    """

    def __init__(self, sets, jj, sq, cu, menu):
        self.total_sets = sets
        self.reps = [(jj, 20), (sq, 35), (cu, 45)]
        self.current_set = 0
        self.progress = 0
        self.current_rep = [0,0,0]

        self.vectors = [np.load("vectors/jj.npy"),np.load("vectors/sq2.npy"),np.load("vectors/cu.npy")]
        self.thresh = [70, 90, 70]

        self.menu = menu

        self.waiting = 0

    def step(self):
        self.current_rep[self.progress] += 1

        self.menu.ui.setValues(
            self.current_set + 1,
            self.current_rep[0],
            self.current_rep[1],
            self.current_rep[2])

        if self.current_rep[self.progress] == self.reps[self.progress][0]:

            self.progress += 1
            if self.progress == 3:
                self.current_set += 1
                if self.current_set != self.total_sets:
                    # Create old window
                    #self.menu.showOldWindow()
                    self.current_rep = [0, 0, 0]
                    # Create Timer
                    self.waiting = 750
                    # Change text
                    self.menu.ui.setToTimer()
                else:
                    self.menu.ui.offTheProgram()

    def reset(self):
        self.progress = 0
        self.menu.ui.setValues(
            self.current_set + 1,
            self.current_rep[0],
            self.current_rep[1],
            self.current_rep[2])
    
    def get_current_avg(self):
        return self.vectors[self.progress], self.thresh[self.progress], self.reps[self.progress][1]

def generate_average_embedding(from_path, total_points, to_path,):
    """
    Function to train model
    """
    points = []
    for i in range(total_points):
        try:
            points.append(np.load(from_path + "{}.npy".format(i)))
        except Exception as e:
            print(e)
    embedder = BodyPoseEmbedding()
    data = np.stack([embedder.get_embeddings(point[0, :, :]) for point in points])
    avg = np.sum(data, axis=0) / data.shape[0]
    np.save(to_path, avg)
