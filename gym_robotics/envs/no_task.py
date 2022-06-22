import copy
import os
from typing import Any, Dict, Optional, Tuple

import gym
import gym.utils
import numpy as np
from gym import error, spaces
from gym_robotics.envs import rotations, utils


try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


class _Renderer:
    """
    Class used for mujoco rendering.
    """

    DEFAULT_SIZE = 500

    def __init__(self, sim: mujoco_py.MjSim) -> None:
        self.sim = sim
        self._viewers = {}
        self.viewer = None  # pytype: mujoco_py.MjViewer

    def render(self, mode: str = "human", width: int = DEFAULT_SIZE, height: int = DEFAULT_SIZE) -> Optional[np.ndarray]:
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode: str) -> mujoco_py.MjViewer:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _viewer_setup(self) -> None:
        self.viewer.cam.lookat[:] = np.array([1.37, 0.73, 0.55])
        self.viewer.cam.distance = 1.0
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}


class FetchNoTaskEnv(gym.Env, gym.utils.EzPickle):
    """
    Fetch environment.
    """

    def __init__(self, image_obs_space: bool = False) -> None:
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "robot0:torso_lift_joint": 0.0,
            "robot0:head_pan_joint": 0.0,
            "robot0:head_tilt_joint": 0.06,
            "robot0:shoulder_pan_joint": 0.01,
            "robot0:shoulder_lift_joint": -0.828,
            "robot0:upperarm_roll_joint": -0.003,
            "robot0:elbow_flex_joint": 1.444,
            "robot0:forearm_roll_joint": 0.003,
            "robot0:wrist_flex_joint": 0.955,
            "robot0:wrist_roll_joint": 0.006,
            "robot0:r_gripper_finger_joint": 0.0,
            "robot0:l_gripper_finger_joint": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }

        self.obj_range = np.array([0.0, 0.0, 0.0])

        model_xml_path = os.path.join(os.path.dirname(__file__), "assets", "fetch", "no_task.xml")
        model = mujoco_py.load_model_from_path(model_xml_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=20)
        self.renderer = _Renderer(self.sim)

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype="float32")
        self.image_obs_space = image_obs_space
        if image_obs_space:
            self.observation_space = spaces.Box(0, 255, (84, 84, 3), dtype="uint8")
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype="float32")
        gym.utils.EzPickle.__init__(self)

    @property
    def dt(self) -> float:
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _env_setup(self, initial_qpos: Dict[str, float]) -> None:
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {}
        reward = 0.0
        return obs, reward, done, info

    def reset(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.seed(seed=seed)
        self._reset_sim()
        obs = self._get_obs()
        return obs

    def _reset_sim(self) -> None:
        self.sim.set_state(self.initial_state)
        # Sample initial object position and set the pose.
        object_pos = np.array([1.3, 0.75, 0.425]) + self.np_random.uniform(-self.obj_range, self.obj_range, size=3)
        object_rot = np.array([0.0, 0.0, 0.0, 1.0])
        object_pos_rot = np.concatenate((object_pos, object_rot))
        self.sim.data.set_joint_qpos("object0:joint", object_pos_rot)
        self.sim.forward()

    def close(self) -> None:
        self.renderer.close()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.renderer.render(mode)

    def _set_action(self, action: np.ndarray) -> None:
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1.0, 0.0, 1.0, 0.0]  # fixed rotation of the end effector, as quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self) -> np.ndarray:
        if self.image_obs_space:
            obs = self.renderer.render("rgb_array", width=84, height=84)
        else:
            # Robot
            gripper_pos = self.sim.data.get_site_xpos("robot0:grip")
            gripper_vel = self.sim.data.get_site_xvelp("robot0:grip") * self.dt
            _robot_pos, _ = utils.robot_get_obs(self.sim)
            gripper_width = _robot_pos[-2:]
            # Object
            object_pos = self.sim.data.get_site_xpos("object0")
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
            object_velp = self.sim.data.get_site_xvelp("object0") * self.dt
            object_velr = self.sim.data.get_site_xvelr("object0") * self.dt

            obs = np.concatenate([gripper_pos, gripper_width, gripper_vel, object_pos, object_rot, object_velp, object_velr], dtype=np.float32)
        return obs.copy()
