#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point

class DroneTargetPublisher(Node):
    def __init__(self):
        super().__init__('drone_target_publisher')

        # Subscribers for SAM and quad odometry
        self.sub_sam  = self.create_subscription(
            Odometry, '/sam_auv_v1/core/odom_gt', self.sam_cb, 10)
        self.sub_quad = self.create_subscription(
            Odometry, '/Quadrotor/odom_gt',      self.quad_cb, 10)

        # Publisher for the relative trajectory
        self.path_pub = self.create_publisher(Path, '/drone/trajectory', 10)

        # Cached poses
        self.sam_pose   = None
        self.quad_pose  = None
        self.start_pose = None   # will be set on first quad_cb

        # Every 0.5 s, regenerate & publish the 20-point path
        self.create_timer(0.5, self.publish_trajectory)

    def sam_cb(self, msg: Odometry):
        self.sam_pose = msg.pose.pose

    def quad_cb(self, msg: Odometry):
        # Cache current quad pose
        self.quad_pose = msg.pose.pose
        # On first callback, record the initial pose
        if self.start_pose is None:
            self.start_pose = msg.pose.pose.position
            self.get_logger().info(
                f"Start pose recorded at "
                f"x={self.start_pose.x:.2f}, "
                f"y={self.start_pose.y:.2f}, "
                f"z={self.start_pose.z:.2f}"
            )

    def publish_trajectory(self):
        # Only run once we have both poses and the start pose
        if not (self.sam_pose and self.quad_pose and self.start_pose):
            return

        # Build a straight-line Path with 20 waypoints
        n = 3
        # start = self.quad_pose.position
        start = Point(
            x=self.sam_pose.position.x + 10.0,
            y=self.sam_pose.position.y,
            z=self.sam_pose.position.z + 10.0
        )

        self.get_logger().info(
                f"Modified Start pose recorded at "
                f"x={start.x:.2f}, "
                f"y={start.y:.2f}, "
                f"z={start.z:.2f}"
            )

        end   = self.sam_pose.position

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        # ‘local’ signals that poses are offsets from start_pose
        path.header.frame_id = 'local'

        for i in range(1, n+1):
            alpha = i / float(n)
            # absolute interpolation
            x_abs = start.x * (1 - alpha) + end.x * alpha
            y_abs = start.y * (1 - alpha) + end.y * alpha
            z_abs = start.z * (1 - alpha) + end.z * alpha

            wp = PoseStamped()
            wp.header = path.header
            # publish the **absolute** world position 
            wp.pose.position.x = x_abs
            wp.pose.position.y = y_abs
            wp.pose.position.z = z_abs

            path.poses.append(wp)
            self.get_logger().info(
                f"Start:   x={self.start_pose.x:.2f}, "
                f"y={self.start_pose.y:.2f}, z={self.start_pose.z:.2f}  →  "
                f"Waypoint: x={wp.pose.position.x:.2f}, "
                f"y={wp.pose.position.y:.2f}, z={wp.pose.position.z:.2f}"
            )


        self.path_pub.publish(path)
        

def main(args=None):
    rclpy.init(args=args)
    node = DroneTargetPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
