#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np

class WaypointFollower(Node):
    def __init__(self):
        """
        Node initialization:
        - Sets up parameters for waypoint generation and publishing rate.
        - Initializes state variables for SAM and quad poses, waypoint list, and step counter.
        - Creates ROS subscriptions for SAM and quad odometry.
        - Creates a publisher for sending waypoint setpoints.
        - Starts a timer to trigger waypoint publishing at fixed intervals.
        """
        super().__init__('waypoint_follower')
        # PARAMETERS
        self.num_steps = 400  # number of waypoints to generate
        self.dt = 0.05        # time interval (seconds) between waypoint publishes
        
        # PARAMETERS FOR TAU BASED TRAJECTORY 
        self.initial_velocity = 5.0      # m/s
        self.tau_k = 0.4                 # shape param k
        self.kd_alpha = 0.8              # α‐coupling exponent
        self.height_offset = 0.4         # final vertical offset [m]
        self.start_threshold  = 0.2      # m to consider “arrived” at start
        
        # STATE
        self.sam_pose = None     # latest SAM position as numpy array
        self.quad_pose = None    # latest quadrotor position as numpy array
        self.waypoints = []      # list of computed waypoints
        self.step_index = 0      # index of the next waypoint to publish

        # Subscribers for SAM and quad odometry
        self.sub_sam = self.create_subscription(
            Odometry, '/sam_auv_v1/core/odom_gt', self.sam_cb, 10)
        self.sub_quad = self.create_subscription(
            Odometry, '/Quadrotor/odom_gt', self.quad_cb, 10)

        # Publisher for waypoint setpoints
        self.wp_pub = self.create_publisher(PoseStamped, '/setpoint_position', 10)
        
        # Timer to call timer_callback at rate 1/dt
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info('WaypointFollower started.')


    def _generate_tau_trajectory(self, p0, p_td):
        delta = p0 - p_td
        d0 = np.linalg.norm(delta)
        if d0 < 1e-6:
            return np.tile(p_td, (self.num_steps,1))

        dir_xy = delta[:2] / np.linalg.norm(delta[:2])
        alpha0 = np.arcsin((p0[2]-p_td[2]) / d0)

        tau0 = -d0 / self.initial_velocity
        t_d  = -tau0 / self.tau_k
        inv_k  = 1.0 / self.tau_k
        inv_kd = 1.0 / self.kd_alpha

        traj = np.zeros((self.num_steps,3))
        for i in range(self.num_steps):
            t = t_d * i / (self.num_steps - 1)
            d = d0 * (1.0 - t/t_d)**inv_k

            alpha = alpha0 * (d/d0)**inv_kd
            cosA, sinA = np.cos(alpha), np.sin(alpha)

            h = d * cosA
            z = p_td[2] + d * sinA

            traj[i,0] = p_td[0] + dir_xy[0] * h
            traj[i,1] = p_td[1] + dir_xy[1] * h
            traj[i,2] = z

        return traj

    def sam_cb(self, msg: Odometry):
        """
        Callback for SAM odometry updates:
        - Extracts position from Odometry message.
        - Stores as numpy array in self.sam_pose.
        """
        p = msg.pose.pose.position
        self.sam_pose = np.array([p.x, p.y, p.z])
        self.get_logger().debug(f'Received SAM pose: {self.sam_pose}')

    def quad_cb(self, msg: Odometry):
        """
        Callback for quadrotor odometry updates:
        - Extracts position from Odometry message.
        - Stores as numpy array in self.quad_pose.
        """
        p = msg.pose.pose.position
        self.quad_pose = np.array([p.x, p.y, p.z])
        self.get_logger().debug(f'Received quad pose: {self.quad_pose}')

    def _compute_waypoints(self, start: np.ndarray, goal: np.ndarray):
        """
        Generate a linear sequence of waypoints from start to goal:
        - Divides the vector from start to goal into num_steps segments.
        - Returns a list of intermediate positions (excluding the start).

        Args:
            start: numpy array [x, y, z] for the starting point.
            goal: numpy array [x, y, z] for the end point.
        Returns:
            List of numpy arrays representing each waypoint.
        """
        return [
            start + (goal - start) * (i / self.num_steps)
            for i in range(1, self.num_steps + 1)
        ]

    def timer_callback(self):
        """
        Periodic function triggered by ROS timer:
        1. Waits until both SAM and quad poses are available.
        2. On first run with both poses, computes the waypoint list from quad to SAM.
        3. Publishes each waypoint in sequence at interval dt.
        4. Shuts down the node when all waypoints have been published.
        """
        # 1. Ensure both poses have been received
        if self.sam_pose is None or self.quad_pose is None:
            self.get_logger().warning('Waiting for both SAM and quad odometry...')
            return

        # 2. Compute waypoints only once, swapping start and goal
        if len(self.waypoints) == 0:  # not self.waypoints: change this logic for V1, V2 trajectory 

            # 1) Define two SAM points: actual and +5 m in X
            pA = self.sam_pose
            pB = pA + np.array([0, 4, 0.0])

            # 2) Midpoint in 3D
            mid3D = (pA + pB) / 2.0

            # 3) Compute 2D perp direction (on x–y plane)
            delta = pB[:2] - pA[:2]
            perp = np.array([-delta[1], delta[0]])
            perp_norm = perp / np.linalg.norm(perp)

            # 4) Offset midpoint by 10 m along perp, and +6 m in Z
            perpendicular_distance = -10.0
            start_height = 7.0
            start2D = mid3D[:2] + perp_norm * perpendicular_distance
            start3D = np.array([
                start2D[0],
                start2D[1],
                mid3D[2] + start_height
            ])
            touchdown = mid3D + np.array([0.0, 0.0, self.height_offset])
           

            self.get_logger().info(
                f'Using start pos {start3D} → goal pos {touchdown}'
            )

            # 5) Precompute straight-line waypoints
            
            self.waypoints = self._generate_tau_trajectory(start3D, touchdown) ## V3 : tau based trajectory from perpendicular starting point
            # self.waypoints = self._compute_waypoints(start3D, mid3D)  ## V2 : straight line approach from perpendicular starting point 
            # self.waypoints = self._compute_waypoints(self.quad_pose, self.sam_pose) ## V1 : straight line approach from random position 
            self.get_logger().info(
                f'Computed {len(self.waypoints)} waypoints from quad to SAM.')

        # 3. Check if we've published all waypoints
        if self.step_index >= len(self.waypoints):
            self.get_logger().info('All waypoints sent. Shutting down node.')
            rclpy.shutdown()
            return

        # Publish next waypoint as a PoseStamped message
        target = self.waypoints[self.step_index]
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(target[0])
        pose_msg.pose.position.y = float(target[1])
        pose_msg.pose.position.z = float(target[2])
        self.wp_pub.publish(pose_msg)

        self.get_logger().info(
            f'Published waypoint {self.step_index + 1}/{len(self.waypoints)}: {target}'
        )
        self.step_index += 1


def main(args=None):
    """
    Entry point for the waypoint follower node:
    - Initializes the ROS client library.
    - Creates the WaypointFollower node instance.
    - Spins to process callbacks until shutdown.
    """
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
