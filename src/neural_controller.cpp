#include "neural_controller/neural_controller.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "controller_interface/helpers.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

namespace neural_controller {
NeuralController::NeuralController()
    : controller_interface::ControllerInterface(),
      rt_command_ptr_(nullptr),
      cmd_subscriber_(nullptr) {}

controller_interface::CallbackReturn NeuralController::on_init() {
  try {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();

    std::ifstream json_stream(params_.model_path, std::ifstream::binary);
    model_ = RTNeural::json_parser::parseJson<float>(json_stream, true);
  } catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn NeuralController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  RCLCPP_INFO(get_node()->get_logger(), "configure successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration NeuralController::command_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::InterfaceConfiguration NeuralController::state_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::CallbackReturn NeuralController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  rt_command_ptr_ = realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>>(nullptr);

  // Populate the command interfaces map
  for (auto &command_interface : command_interfaces_) {
    command_interfaces_map_[command_interface.get_prefix_name()].emplace(
        command_interface.get_interface_name(), std::ref(command_interface));
  }

  // Populate the state interfaces map
  for (auto &state_interface : state_interfaces_) {
    state_interfaces_map_[state_interface.get_prefix_name()].emplace(
        state_interface.get_interface_name(), std::ref(state_interface));
  }

  // Store the initial joint positions
  for (int i = 0; i < ACTION_SIZE; i++) {
    init_joint_pos_.at(i) =
        state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value();
  }

  init_time_ = get_node()->now();
  repeat_action_counter_ = -1;

  cmd_x_vel_ = 0.0;
  cmd_y_vel_ = 0.0;
  cmd_yaw_vel_ = 0.0;

  // Set the gravity z-component in the initial observation vector
  for (int i = 0; i < OBSERVATION_HISTORY; i++) {
    observation_.at(i * SINGLE_OBSERVATION_SIZE + OBSERVATION_GRAVITY_Z_INDEX) = -1.0;
  }

  // Initialize the command subscriber
  cmd_subscriber_ = get_node()->create_subscription<CmdType>(
      "/cmd_vel", rclcpp::SystemDefaultsQoS(),
      [this](const CmdType::SharedPtr msg) { rt_command_ptr_.writeFromNonRT(msg); });

  RCLCPP_INFO(get_node()->get_logger(), "activate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn NeuralController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  rt_command_ptr_ = realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>>(nullptr);
  for (auto &command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  RCLCPP_INFO(get_node()->get_logger(), "deactivate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type NeuralController::update(const rclcpp::Time &time,
                                                           const rclcpp::Duration &period) {
  // When started, return to the default joint positions
  double time_since_init = (time - init_time_).seconds();
  if (time_since_init < params_.init_duration) {
    for (int i = 0; i < ACTION_SIZE; i++) {
      // Interpolate between the initial joint positions and the default joint
      // positions
      double interpolated_joint_pos =
          init_joint_pos_.at(i) * (1 - time_since_init / params_.init_duration) +
          params_.default_joint_pos.at(i) * (time_since_init / params_.init_duration);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("position")
          .get()
          .set_value(interpolated_joint_pos);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kp")
          .get()
          .set_value(params_.init_kps.at(i));
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kd")
          .get()
          .set_value(params_.init_kds.at(i));
    }
    return controller_interface::return_type::OK;
  }

  // After the init_duration has passed, fade in the policy actions
  double time_since_fade_in = (time - init_time_).seconds() - params_.init_duration;
  float fade_in_multiplier = std::min(time_since_fade_in / params_.fade_in_duration, 1.0);

  // If an emergency stop has been triggered, set all commands to 0 and return
  if (estop_active_) {
    for (auto &command_interface : command_interfaces_) {
      command_interface.set_value(0.0);
    }
    return controller_interface::return_type::OK;
  }

  // Only get a new action from the policy when repeat_action_counter_ is 0
  repeat_action_counter_ += 1;
  repeat_action_counter_ %= params_.repeat_action;
  if (repeat_action_counter_ != 0) {
    return controller_interface::return_type::OK;
  }

  // Get the latest commanded velocities
  auto command = rt_command_ptr_.readFromRT();
  if (command && command->get()) {
    cmd_x_vel_ = command->get()->linear.x;
    cmd_y_vel_ = command->get()->linear.y;
    cmd_yaw_vel_ = command->get()->angular.z;
  }

  // Get the latest observation
  double ang_vel_x, ang_vel_y, ang_vel_z, orientation_w, orientation_x, orientation_y,
      orientation_z;
  try {
    // read IMU states from hardware interface
    ang_vel_x = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.x")
                    .get()
                    .get_value();
    ang_vel_y = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.y")
                    .get()
                    .get_value();
    ang_vel_z = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.z")
                    .get()
                    .get_value();
    orientation_w =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.w").get().get_value();
    orientation_x =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.x").get().get_value();
    orientation_y =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.y").get().get_value();
    orientation_z =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.z").get().get_value();

    // Calculate the projected gravity vector
    tf2::Quaternion q(orientation_x, orientation_y, orientation_z, orientation_w);
    tf2::Matrix3x3 m(q);
    tf2::Vector3 world_gravity_vector(0, 0, -1);
    tf2::Vector3 projected_gravity_vector = m.inverse() * world_gravity_vector;

    // If the maximum body angle is exceeded, trigger an emergency stop
    if (-projected_gravity_vector[2] < cos(params_.max_body_angle)) {
      estop_active_ = true;
      RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      return controller_interface::return_type::OK;
    }

    // Fill the observation vector
    // Angular velocity
    observation_.at(0) = ang_vel_x * params_.ang_vel_scale;
    observation_.at(1) = ang_vel_y * params_.ang_vel_scale;
    observation_.at(2) = ang_vel_z * params_.ang_vel_scale;
    // Projected gravity vector
    observation_.at(3) = projected_gravity_vector[0];
    observation_.at(4) = projected_gravity_vector[1];
    observation_.at(5) = projected_gravity_vector[2];
    // Commands
    observation_.at(6) = cmd_x_vel_ * params_.lin_vel_scale;
    observation_.at(7) = cmd_y_vel_ * params_.lin_vel_scale;
    observation_.at(8) = cmd_yaw_vel_ * params_.ang_vel_scale;
    // Joint positions
    for (int i = 0; i < ACTION_SIZE; i++) {
      // Only include the joint position in the observation if the action type
      // is position
      if (params_.action_types.at(i) == "position") {
        observation_.at(SENSOR_OBSERVATION_SIZE + i) =
            (state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value() -
             params_.default_joint_pos.at(i)) *
            params_.joint_pos_scale;
      }
    }
  } catch (const std::out_of_range &e) {
    RCLCPP_INFO(get_node()->get_logger(), "failed to read joint states from hardware interface");
    return controller_interface::return_type::OK;
  }

  // Clip the observation vector
  for (int i = 0; i < OBSERVATION_SIZE; i++) {
    observation_.at(i) = std::max(std::min(observation_.at(i), (float)params_.observation_limit),
                                  (float)-params_.observation_limit);
  }

  // Perform policy inference
  model_->forward(observation_.data());

  // Shift the observation history to the right by SINGLE_OBSERVATION_SIZE for the next control step
  // https://en.cppreference.com/w/cpp/algorithm/rotate
  std::rotate(observation_.rbegin(), observation_.rbegin() + SINGLE_OBSERVATION_SIZE,
              observation_.rend());

  // Process the actions
  std::array<float, ACTION_SIZE> policy_output;
  const float *policy_output_raw = model_->getOutputs();
  std::copy(policy_output_raw, policy_output_raw + ACTION_SIZE, policy_output.begin());

  for (int i = 0; i < ACTION_SIZE; i++) {
    // Clip the action
    float action = fade_in_multiplier * policy_output.at(i) * params_.action_scales.at(i);
    if (params_.action_types.at(i) == "position") {
      action = action + params_.default_joint_pos.at(i);
    }
    float clipped_action = std::max(std::min(action, (float)params_.action_limit_upper.at(i)),
                                    (float)params_.action_limit_lower.at(i));
    action_.at(i) = clipped_action;

    // Copy policy_output to the observation vector
    observation_.at(SENSOR_OBSERVATION_SIZE + ACTION_SIZE + i) =
        fade_in_multiplier * policy_output.at(i);

    // Send the action to the hardware interface
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at(params_.action_types.at(i))
        .get()
        .set_value((double)action_.at(i));
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kp")
        .get()
        .set_value(params_.kps.at(i));
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(params_.kds.at(i));
  }

  // Get the policy inference time
  // double policy_inference_time = (get_node()->now() - time).seconds();
  // RCLCPP_INFO(get_node()->get_logger(), "policy inference time: %f",
  // policy_inference_time);

  return controller_interface::return_type::OK;
}

}  // namespace neural_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(neural_controller::NeuralController,
                       controller_interface::ControllerInterface)