#ifndef NEURAL_CONTROLLER__NEURAL_CONTROLLER_HPP_
#define NEURAL_CONTROLLER__NEURAL_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <functional>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.h"
#include "realtime_tools/realtime_publisher.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "geometry_msgs/msg/twist.hpp"

#include <RTNeural/RTNeural.h>

// auto-generated by generate_parameter_library
#include "neural_controller_parameters.hpp"

namespace neural_controller
{
  using CmdType = geometry_msgs::msg::Twist;

  class NeuralController 
  : public controller_interface::ControllerInterface
  {
  public:
    NeuralController();

    ~NeuralController() = default;

    controller_interface::InterfaceConfiguration command_interface_configuration() const override;

    controller_interface::InterfaceConfiguration state_interface_configuration() const override;

    controller_interface::CallbackReturn on_init() override;

    controller_interface::CallbackReturn on_configure(
        const rclcpp_lifecycle::State &previous_state) override;

    controller_interface::CallbackReturn on_activate(
        const rclcpp_lifecycle::State &previous_state) override;

    controller_interface::CallbackReturn on_deactivate(
        const rclcpp_lifecycle::State &previous_state) override;

    controller_interface::return_type update(
        const rclcpp::Time &time, const rclcpp::Duration &period) override;

  protected:
    /* ----------------- Layer sizes ----------------- */
    static constexpr int ACTION_SIZE = 6;
    static constexpr int OBSERVATION_SIZE = 3 /* base link linear velocity */\
      + 3 /* base link angular velocity */\
      + 3 /* projected gravity vector */\
      + 3 /* x, y, yaw velocity commands */\
      + ACTION_SIZE /* joint positions */\
      + ACTION_SIZE /* joint velocities */\
      + ACTION_SIZE; /* previous action */
    static constexpr int LSTM_SIZE = 512;
    static constexpr int DENSE_0_SIZE = 512;
    static constexpr int DENSE_1_SIZE = 256;
    static constexpr int DENSE_2_SIZE = 128;
    /* ----------------------------------------------- */

    /* ----------------- Model definition ----------------- */
    RTNeural::ModelT<float, OBSERVATION_SIZE, ACTION_SIZE,
      RTNeural::LSTMLayerT<float, OBSERVATION_SIZE, LSTM_SIZE>,
      RTNeural::DenseT<float, LSTM_SIZE, DENSE_0_SIZE>,
      RTNeural::ELuActivationT<float, DENSE_0_SIZE>,
      RTNeural::DenseT<float, DENSE_0_SIZE, DENSE_1_SIZE>,
      RTNeural::ELuActivationT<float, DENSE_1_SIZE>,
      RTNeural::DenseT<float, DENSE_1_SIZE, DENSE_2_SIZE>,
      RTNeural::ELuActivationT<float, DENSE_2_SIZE>,
      RTNeural::DenseT<float, DENSE_2_SIZE, ACTION_SIZE>
    > model_;
    /* ----------------------------------------------------- */

    std::shared_ptr<ParamListener> param_listener_;
    Params params_;

    float observation_[OBSERVATION_SIZE];
    float action_[ACTION_SIZE];

    float cmd_x_vel_, cmd_y_vel_, cmd_yaw_vel_;

    // Map from joint names to command types to command interfaces
    std::map<std::string, std::map<std::string, std::reference_wrapper<hardware_interface::LoanedCommandInterface>>>
        command_interfaces_map_;

    // Map from joint/sensor names to state types to state interfaces
    std::map<std::string, std::map<std::string, std::reference_wrapper<hardware_interface::LoanedStateInterface>>>
        state_interfaces_map_;

    realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>> rt_command_ptr_;
    rclcpp::Subscription<CmdType>::SharedPtr cmd_subscriber_;

    rclcpp::Time init_time_;

    float wrap_angle(float angle, float angle_min, float angle_max);

  };

} // namespace neural_controller

#endif // NEURAL_CONTROLLER__NEURAL_CONTROLLER_HPP_