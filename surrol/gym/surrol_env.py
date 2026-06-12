import time
import socket
import os
import threading

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data
import pkgutil
from surrol.utils.pybullet_utils import (
    step,
    render_image,
)
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

RENDER_HEIGHT = 480  # train
RENDER_WIDTH = 640
# RENDER_HEIGHT = 1080  # record
# RENDER_WIDTH = 1920


class SurRoLEnv(gym.Env):
    """
    A gym Env wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'img_array'],
        'render_modes': ['human', 'rgb_array', 'img_array'],
    }

    def __init__(self, render_mode: str = None):
        # rendering and connection options
        self._render_mode = render_mode
        # render_mode = 'human'
        # if render_mode == 'human':
        #     self.cid = p.connect(p.SHARED_MEMORY)
        #     if self.cid < 0:
        if render_mode == 'human':
            self.cid = p.connect(p.GUI)
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.cid = p.connect(p.DIRECT)
            # See PyBullet Quickstart Guide Synthetic Camera Rendering
            # TODO: no light when using direct without egl
            disable_egl = os.environ.get("SURROL_DISABLE_EGL", "0") == "1"
            if socket.gethostname().startswith('pc') and not disable_egl:
                egl = pkgutil.get_loader('eglRenderer')
                if egl is not None:
                    try:
                        p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                    except Exception:
                        # Safe fallback for headless/unsupported EGL
                        pass
        # camera related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(0, 0, 0.2),
                                                                distance=1.5,
                                                                yaw=90,
                                                                pitch=-36,
                                                                roll=0,
                                                                upAxisIndex=2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov=45,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=20.0)
        # additional settings
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        # Disable side-previews to fix flickering
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf", (0, 0, -0.001))
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        self.seed()
        self._duration = 0.2  # important for mini-steps

        # self.actions = []  # only for demo
        self._render_lock = threading.Lock()
        self._last_render_time = 0
        
        # Dual-view parameters for tuning
        self._top_down_distance = 0.18  # Zoom in (smaller value = larger objects)
        self._top_down_target = None   # Will be set based on scaling in render()

        self._env_setup()
        step(0.25)
        self.goal = self._sample_goal()  # tasks are all implicitly goal-based
        self._sample_goal_callback()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(self.action_size,), dtype='float32')
        if isinstance(obs, np.ndarray):
            # gym.Env
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
        elif isinstance(obs, dict):
            # gym.GoalEnv
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ))
        else:
            raise NotImplementedError

        if self._render_mode == 'human':
            self._cv2_init_done = False

    def step(self, action: np.ndarray):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # time0 = time.time()
        self._set_action(action)
        # time1 = time.time()
        # TODO: check the best way to step simulation
        step(self._duration)
        if self._render_mode == 'human':
            self.render(mode='human')

        # time2 = time.time()
        # print(" -> robot action time: {:.6f}, simulation time: {:.4f}".format(time1 - time0, time2 - time1))
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } if isinstance(obs, dict) else {'achieved_goal': None}
        if isinstance(obs, dict):
            reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        else:
            reward = self.compute_reward(obs, self.goal, info)
        # if len(self.actions) > 0:
        #     self.actions[-1] = np.append(self.actions[-1], [reward])  # only for demo
        return obs, reward, done, info

    def reset(self):
        # reset scene in the corresponding file
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

        p.loadURDF("plane.urdf", (0, 0, -0.001))
        self._env_setup()
        step(0.25)
        self.goal = self._sample_goal().copy()
        self._sample_goal_callback()

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        if self._render_mode == 'human':
            self.render(mode='human')

        obs = self._get_obs()
        return obs

    def close(self):
        """
        Safely disconnect from the physics engine and cleanup GUI resources.
        """
        # Synchronize cleanup using the render lock
        with self._render_lock:
            if self.cid >= 0:
                try:
                    p.disconnect(self.cid)
                except Exception:
                    # Fallback if cid is already invalid or disconnect fails
                    try:
                        p.disconnect()
                    except:
                        pass
                self.cid = -1
                
            if cv2 is not None:
                try:
                    # Clear init flag and destroy windows in a safe state
                    if hasattr(self, '_cv2_init_done'):
                        self._cv2_init_done = False
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def render(self, mode='rgb_array'):
        # Rate limit and thread-safety to avoid flickering
        now = time.time()
        
        # Use non-blocking acquire to avoid overlapping render calls if one is slow.
        if not self._render_lock.acquire(blocking=False):
            if (self._render_mode == 'human' or mode == 'human') and cv2 is not None:
                cv2.waitKey(1)
            return np.array([]) if mode == 'human' else (np.array([]), None)
            
        try:
            # 25 FPS limit for 'human' mode to reduce CPU load and prevent GUI congestion
            if mode == 'human' or self._render_mode == 'human':
                if now - self._last_render_time < 0.04:
                    if getattr(self, '_cv2_init_done', False) and cv2 is not None:
                        cv2.waitKey(1)
                    return np.array([]) if mode == 'human' else (np.array([]), None)
                self._last_render_time = now

            # 【SYNC FIX】: Dynamically synchronize with the PyBullet GUI camera.
            # This ensures the OpenCV window follows mouse rotations/translations in the PyBullet window.
            if self._render_mode == 'human':
                try:
                    cam_info = p.getDebugVisualizerCamera()
                    if len(cam_info) >= 12:
                        # Index 2: viewMatrix, Index 3: projectionMatrix
                        self._view_matrix = cam_info[2]
                        self._proj_matrix = cam_info[3]
                        
                        # Fallback/alternative: Reconstruct view matrix if needed
                        # self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
                        #     cameraTargetPosition=cam_info[11], distance=cam_info[10],
                        #     yaw=cam_info[8], pitch=cam_info[9], roll=0, upAxisIndex=2)
                except Exception:
                    pass

            self._render_callback(mode)
            
            # Main view - uses synced matrices from GUI
            rgb_array, mask = render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                           self._view_matrix, self._proj_matrix)
            
            if rgb_array is None or rgb_array.size == 0:
                return np.array([]) if mode == 'human' else (np.array([]), None)

            # Fixed Top-down view
            scaling = getattr(self, 'SCALING', 1.0)
            fixed_target = (0.55 * scaling, 0, 0.68 * scaling)
            fixed_distance = self._top_down_distance * scaling
            
            top_view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=fixed_target,
                distance=fixed_distance,
                yaw=90,
                pitch=-89.9,
                roll=0,
                upAxisIndex=2
            )
            # Use a fixed projection matrix for top-down to avoid distortion from GUI aspect ratio
            top_proj_matrix = p.computeProjectionMatrixFOV(
                fov=45, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                nearVal=0.1, farVal=20.0
            )
            
            # Render top-down. Use Tiny Renderer if hardware OpenGL is busy/flickering
            top_rgb, _ = render_image(RENDER_WIDTH // 2, RENDER_HEIGHT // 2, 
                                      top_view_matrix, top_proj_matrix)
            
            if top_rgb is not None and top_rgb.size > 0:
                top_rgb = cv2.resize(top_rgb, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
                combined_rgb = np.concatenate([rgb_array, top_rgb], axis=1)
            else:
                combined_rgb = rgb_array

            # Show dual-view window
            if (self._render_mode == 'human' or mode == 'human') and cv2 is not None:
                win_name = "SurRoL Dual-View (Main | Top-down)"
                try:
                    if not getattr(self, '_cv2_init_done', False):
                        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
                        if hasattr(cv2, 'startWindowThread'):
                            cv2.startWindowThread()
                        self._cv2_init_done = True
                    
                    bgr_image = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(win_name, bgr_image)
                    # Use a slightly longer waitKey or explicit event pump
                    cv2.waitKey(1)
                except Exception:
                    pass

            if mode == 'rgb_array':
                return combined_rgb
            elif mode == 'human':
                return np.array([])
            else:
                return combined_rgb, mask
        finally:
            self._render_lock.release()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def _env_setup(self):
        pass

    def _get_obs(self):
        raise NotImplementedError

    def _set_action(self, action):
        """ Applies the given action to the simulation.
        """
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError

    def _sample_goal(self):
        """ Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _sample_goal_callback(self):
        """ For goal visualization, etc.
        """
        pass

    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """ A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    @property
    def action_size(self):
        raise NotImplementedError

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a scripted oracle strategy
        """
        raise NotImplementedError

    def test(self, horizon=100):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        steps, done = 0, False
        obs = self.reset()
        while not done and steps <= horizon:
            tic = time.time()
            action = self.get_oracle_action(obs)
            print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
            # print('action:', action)
            obs, reward, done, info = self.step(action)
            if isinstance(obs, dict):
                print(" -> achieved goal: {}".format(np.round(obs['achieved_goal'], 4)))
                print(" -> desired goal: {}".format(np.round(obs['desired_goal'], 4)))
            else:
                print(" -> achieved goal: {}".format(np.round(info['achieved_goal'], 4)))
            done = info['is_success'] if isinstance(obs, dict) else done
            steps += 1
            toc = time.time()
            print(" -> step time: {:.4f}".format(toc - tic))
            time.sleep(0.05)
        print('\n -> Done: {}\n'.format(done > 0))

    def __del__(self):
        self.close()
