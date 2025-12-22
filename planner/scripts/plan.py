import sys
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from utils import traj2ros
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config

class PCTPlanner(Node):
    def __init__(self, cfg, tomo_file, start_pos, end_pos):
        super().__init__('pct_planner')
        self.cfg = cfg
        self.tomo_file = tomo_file
        self.start_pos = start_pos
        self.end_pos = end_pos

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_pub = self.create_publisher(Path, "/pct_path", qos)
        self.cluster_pub = self.create_publisher(MarkerArray, '/path_markers', qos)
        
        # 订阅 Rviz 的 2D Goal Pose 工具 (通常发布在 /goal_pose)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # 订阅 Rviz 的 Publish Point 工具 (通常发布在 /clicked_point)
        self.point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.point_callback,
            10
        )

        self.planner = TomogramPlanner(cfg)
        self.planner.loadTomogram(self.tomo_file) # 只加载一次地图

        # 修正初始坐标：假设 plan.py 中硬编码的 z 是层级索引，我们需要将其转换为物理高度
        # 这样才能与 Rviz 点击的物理高度逻辑保持一致，并适配 planner_wrapper 的计算公式
        if self.start_pos.shape[0] == 3:
            layer_idx = self.start_pos[2]
            self.start_pos[2] = layer_idx * self.planner.slice_dh + self.planner.slice_h0
            self.get_logger().info(f"Converted Start Pos Z from Layer {layer_idx} to Height {self.start_pos[2]:.2f}")

        if self.end_pos.shape[0] == 3:
            layer_idx = self.end_pos[2]
            self.end_pos[2] = layer_idx * self.planner.slice_dh + self.planner.slice_h0
            self.get_logger().info(f"Converted End Pos Z from Layer {layer_idx} to Height {self.end_pos[2]:.2f}")

        self.pct_plan()

    def goal_callback(self, msg):
        """
        当在 Rviz 中使用 '2D Goal Pose' 工具点击时触发。
        更新终点并重新规划。
        注意：Rviz 的 2D Goal Pose 默认 z=0。
        如果需要 3D 终点，可能需要自定义 Rviz 插件或通过其他方式发布目标点。
        这里我们假设用户点击的是当前层级，或者保持原有的 z 高度。
        """
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        
        # 策略：保持原有的 z 高度 (层级)，只更新 x, y
        # 或者，如果您的场景是平面的，z 可以设为 0
        current_z = self.end_pos[2] if self.end_pos.shape[0] > 2 else 0.0
        
        self.get_logger().info(f"Received new goal (2D Pose): x={new_x:.2f}, y={new_y:.2f}. Re-planning...")
        
        if self.end_pos.shape[0] == 3:
             self.end_pos = np.array([new_x, new_y, current_z], dtype=np.float32)
        else:
             self.end_pos = np.array([new_x, new_y], dtype=np.float32)

        self.pct_plan()

    def point_callback(self, msg):
        """
        当在 Rviz 中使用 'Publish Point' 工具点击时触发。
        支持 3D 坐标输入。
        """
        new_x = msg.point.x
        new_y = msg.point.y
        new_z = msg.point.z

        self.get_logger().info(f"Received new goal (3D Point): x={new_x:.2f}, y={new_y:.2f}, z={new_z:.2f}. Re-planning...")

        if self.end_pos.shape[0] == 3:
             self.end_pos = np.array([new_x, new_y, new_z], dtype=np.float32)
        else:
             # 如果原本是2D模式，这里强制升级为3D模式以支持Z轴
             self.end_pos = np.array([new_x, new_y, new_z], dtype=np.float32)
        
        self.pct_plan()

    def publish_cluster_marker(self):
        marker_array = MarkerArray()
        for i, pos in enumerate([self.start_pos, self.end_pos]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cluster"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2]) if len(pos) > 2 else 0.5
            marker_array.markers.append(marker)
        self.cluster_pub.publish(marker_array)

    def pct_plan(self):
        self.publish_cluster_marker()
        traj_3d = self.planner.plan(self.start_pos, self.end_pos)
        if traj_3d is not None:
            self.path_pub.publish(traj2ros(traj_3d))
            self.get_logger().info("Trajectory published")
        else:
            self.get_logger().warn("Failed to find a path!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='Spiral', help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
    args = parser.parse_args()

    cfg = Config()

    match args.scene:
        case 'Spiral':
            tomo_file = 'spiral0.3_2'
            start_pos = np.array([-16.0, -6.0], dtype=np.float32)
            end_pos = np.array([-26.0, -5.0], dtype=np.float32)
        case 'Building':
            tomo_file = 'building'
            start_pos = np.array([-5.29, 6.01, 0.06], dtype=np.float32)
            end_pos = np.array([4.21, 2.81, 7], dtype=np.float32)
        case 'Plaza':
            tomo_file = 'plaza3_10'
            start_pos = np.array([0.0, 0.0], dtype=np.float32)
            end_pos = np.array([23.0, 10.0], dtype=np.float32)
        case _:
            raise ValueError('Invalid scene name')

    rclpy.init(args=None)

    node = PCTPlanner(cfg, tomo_file, start_pos, end_pos)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()