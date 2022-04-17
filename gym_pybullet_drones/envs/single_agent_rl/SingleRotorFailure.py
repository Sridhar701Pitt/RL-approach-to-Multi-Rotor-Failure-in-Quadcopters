import numpy as np
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

from gym import spaces
import pybullet as p
class SingleRotorFailure(HoverAviary):
    def __init__(
            self,
            drone_model=DroneModel.CF2X,
            initial_xyzs=np.array([[0.0,0.0,0.0]]),
            initial_rpys=None,
            physics=Physics.PYB,
            freq: int = 240,
            aggregate_phy_steps: int = 1,
            gui=False,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM
    ):
        self.goal_point = np.array([0.0, 0.0, 0.0])
        self.GUI = gui
        if self.GUI:
            ## Drone tracking params
            self.rayFrom = []
            self.rayTo = []
            self.rayIds = []
            self.renderCount = 0
            ## goal tracking params
            self.gRayFrom = []
            self.gRayTo = []
            self.gRayIds = []
            
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.initial_point = initial_xyzs
        self.goal_point = np.array([0.0, 0.0, 0.0])




    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        #EDIT: relative position with respect to the goal point
        relative_pos_xy = state[0:2] - self.goal_point[0:2]
        relative_pos_z = state[2] - self.goal_point[2]

        clipped_pos_xy = np.clip(relative_pos_xy, -MAX_XY, MAX_XY) #EDIT: relative xy positons
        clipped_pos_z = np.clip(relative_pos_z, -MAX_Z, MAX_Z) #EDIT: z position can be negative and is relative
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped

        ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8
                              )
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 12
            return spaces.Box(low=np.array([-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1]), #EDIT: z spans from -1 to 1
                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]),
                              dtype=np.float32
                              )
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._observationSpace()")
    
    ################################################################################

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        # We only ever consider RPMS and set the 4th RPM to 0 to simulate rotor failure
        rpms = super()._preprocessAction(action)
        # Set 1,3rd motor value to 0
        # dual rotor
        # rpms[0] = 0
        rpms[2] = 0
        # print(action)
        # print(rpms)
        return rpms

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        if self.GUI == True and len(self.gRayFrom) != 0:
            # If custom goalpoints
            self.INIT_XYZS = self.initial_point
        else:
            ## WE want to introduce some variation in the intial positions for robustness
            self.INIT_XYZS = self.initial_point + 1.0*(2*np.random.rand(1,3)-1)

        if self.GUI:
            #EDIT: update drone tracer
            if len(self.rayFrom) != 0:
                #Not empty list
                self.rayFrom.pop(-1)
            self.rayFrom.append(self.INIT_XYZS.flatten())

            # Reset goal
            if len(self.gRayFrom) != 0:
                self.goal_point = self.gRayTo[0]
                print("Initial Goal (rays): ", self.goal_point)
            else:
                print("Initial Goal (not rays): ", self.goal_point)

        return super().reset()
    
    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        if self.GUI:
            ## Add some lines oohooo
            # Draw goal path
            if len(self.gRayFrom) != 0:
                for i in range(len(self.gRayFrom)):
                    self.gRayIds.append(p.addUserDebugLine(self.gRayFrom[i], self.gRayTo[i], lineColorRGB=[1.0,0.1,0.1],lineWidth=2, physicsClientId=self.CLIENT))
            else:
                p.addUserDebugLine(self.INIT_XYZS.flatten(),self.goal_point,lineColorRGB=[1.0,0.1,0.1],lineWidth=2, physicsClientId=self.CLIENT)

        return super()._housekeeping()

    ################################################################################

    def _computeInfo(self): #This functions executes near the end of step function
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """

        ## EDIT: update drone tracers
        if self.GUI:
            i = self.renderCount
            xyz_positions = self._getDroneStateVector(0)[0:3]
            self.rayFrom.append(xyz_positions)
            self.rayTo.append(xyz_positions)
            self.rayIds.append(p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], lineColorRGB=[0.1,0.1,1.0],lineWidth=2, physicsClientId=self.CLIENT))
            self.renderCount = self.renderCount + 1

            if len(self.gRayFrom) != 0:
                # Update goal point and change color
                self.goal_point = self.gRayTo[self.renderCount]
                p.addUserDebugLine(self.gRayFrom[i],self.gRayTo[i], lineColorRGB=[0.1,1.0,0.1],replaceItemUniqueId=self.gRayIds[i],lineWidth=2, physicsClientId=self.CLIENT)


        return super()._computeInfo()

    ################################################################################
    
    def injectGoals(self, goal_path):
        if self.GUI:
            self.initial_point = goal_path[0].reshape((1,3))
            self.gRayFrom = goal_path[0:-1]
            self.gRayTo = goal_path[1:]

