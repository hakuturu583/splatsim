"""Launch splatsim spawn-scenario with shutdown-on-exit."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("carla_host", default_value="localhost"),
        DeclareLaunchArgument("carla_port", default_value="2000"),
        DeclareLaunchArgument("scene_yaml"),
        DeclareLaunchArgument("spawn_lanelet_id", default_value="2303321"),
        DeclareLaunchArgument("spawn_s", default_value="0.0"),
        DeclareLaunchArgument("vehicle_type", default_value="vehicle.mini.cooper_s_2021"),
        DeclareLaunchArgument("timeout_seconds", default_value="3600.0"),
        DeclareLaunchArgument("image_topic", default_value="/splatsim/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/splatsim/camera_info"),
        DeclareLaunchArgument("frame_id", default_value="splatsim_camera"),
    ]

    spawn_scenario = ExecuteProcess(
        cmd=[
            PathJoinSubstitution([FindPackageShare("splatsim"), "..", "..", "lib", "splatsim", "spawn-scenario"]),
            ["server.host=", LaunchConfiguration("carla_host")],
            ["server.port=", LaunchConfiguration("carla_port")],
            ["splatsim.scene_yaml=", LaunchConfiguration("scene_yaml")],
            ["ego.spawn_lanelet_id=", LaunchConfiguration("spawn_lanelet_id")],
            ["ego.spawn_s=", LaunchConfiguration("spawn_s")],
            ["ego.vehicle_type=", LaunchConfiguration("vehicle_type")],
            ["scenario.timeout_seconds=", LaunchConfiguration("timeout_seconds")],
            ["splatsim.image_topic=", LaunchConfiguration("image_topic")],
            ["splatsim.camera_info_topic=", LaunchConfiguration("camera_info_topic")],
            ["splatsim.frame_id=", LaunchConfiguration("frame_id")],
        ],
        name="spawn-scenario",
        output="screen",
    )

    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_scenario,
            on_exit=[
                TimerAction(
                    period=3.0,
                    actions=[EmitEvent(event=Shutdown(reason="spawn-scenario exited"))],
                )
            ],
        )
    )

    return LaunchDescription([*args, spawn_scenario, shutdown_handler])
