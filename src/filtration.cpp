#include <ros/ros.h>
#include <ros/package.h>

#include <mrs_msgs/PoseWithCovarianceArrayStamped.h>
#include <mrs_msgs/PoseWithCovarianceIdentified.h>
#include <mrs_lib/transformer.h>
#include <mrs_lib/param_loader.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <boost/bind.hpp>
#include <mrs_lib/ukf.h>
#include <deque>
#include <Eigen/Dense>
#include <std_srvs/Trigger.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>

namespace filtration {

  struct VisualFieldParams {
    double GAM;
    double V0;
    double ALP0;
    double ALP1;
    double ALP2;
    double BET0;
    double BET1;
    double BET2;
    double R;
    int field_size;
  };

  struct Position {
    double x, y, z, heading;
  };

  // Define UKF types
  const int n_states = 10; // x, y, z, vx, vy, vz, roll, pitch, yaw, v
  const int n_inputs = 0;  // No explicit inputs
  const int n_measurements = 6; // x, y, z, roll, pitch, yaw
  using ukf_t = mrs_lib::UKF<n_states, n_inputs, n_measurements>;
  using x_t = ukf_t::x_t;
  using z_t = ukf_t::z_t;
  using Q_t = ukf_t::Q_t;
  using R_t = ukf_t::R_t;
  using statecov_t = ukf_t::statecov_t;
  
  class PoseFiltration {
    private:
      ros::NodeHandle nh_;
      std::vector<ros::Subscriber> uvdar_subscribers_;
      std::vector<ros::Publisher> filtered_pose_publishers_;
      std::vector<std::string> uav_names;
      ros::Time last_timestamp;
      ros::Subscriber position_subscriber;
      std::vector<ros::Subscriber> odom_subscribers;
      std::vector<Eigen::Vector3d> odom_positions;
      std::vector<Eigen::Vector3d> uav_positions;

      // UKF-specific variables
      std::unordered_map<int, std::unordered_map<int, statecov_t>> ukf_map_;
      Q_t process_noise_;
      ukf_t::transition_model_t transition_model_;
      ukf_t::observation_model_t observation_model_;
      ukf_t ukf_;
      int current_agent_id_; 
      int field_size_; 
      std::vector<x_t> all_states_;
      std::vector<Position> initial_positions;
      double q_vel;
      double q_pos;
      double q_ori;
      double last_callback_time_;
      
      // activation service
      ros::ServiceServer service_activate_filtration_;
      
      // fish model variables
      VisualFieldParams params;
      bool filtration_allowed_ = false;

      std::ofstream predicted_file_, measured_file_, groundtruth_file_;
      bool files_initialized_ = false;
      double timestamp;

		  mrs_lib::Transformer transformer_;

    public:
      PoseFiltration(ros::NodeHandle& nh) : nh_(nh),
                                            transition_model_([&](const x_t& state, const ukf_t::u_t& input, const double dt) {
                                              return transitionModel(state, input, dt, current_agent_id_, params, all_states_);
                                            }),
                                            observation_model_(observationModel),
                                            ukf_(transition_model_, observation_model_, 1e-3, 1, 2) {
        loadParameters();
        initializeSubscribersAndPublishers();
        setupProcessNoise();
        initializeFiles();
        last_callback_time_ = getCurrentTimeAsDouble();
        transformer_ = mrs_lib::Transformer("FiltrationTransformer");
        service_activate_filtration_ = nh_.advertiseService("filtration_activation_in", &PoseFiltration::activationServiceCallback, this);
      }

      ~PoseFiltration() {
        closeFiles();
      }

          // Transition model
      x_t transitionModel(const x_t& state, const ukf_t::u_t& input, const double dt, int current_agent_id_, const VisualFieldParams& params, std::vector<x_t> all_states_) {
        x_t next_state = state;

        // Extract current state variables
        double x = state(0), y = state(1), z = state(2), psi = state(8), v = state(9);
        double heading = atan2(state(4), state(3));
        Eigen::VectorXd visual_field = compute_visual_field(heading, all_states_, params, current_agent_id_);
        // int vis_field_sum = visual_field.sum();
        Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(params.field_size, -M_PI, M_PI);
        auto [dvel, dpsi] = compute_state_variables(v, phi, visual_field, params);
        // Update velocity and heading
        auto new_v = v + dvel * dt;
        next_state(9) = new_v;
        next_state(8) = psi;
        heading += dpsi * dt;
        auto new_vx = new_v * std::cos(heading);
        auto new_vy = new_v * std::sin(heading);
        next_state(3) = new_vx; // vx
        next_state(4) = new_vy; // vy
        next_state(5) = state(5); // vz
        // Position updates
        next_state(0) = x + dt * new_vx; // x
        next_state(1) = y + dt * new_vy; // y
        next_state(2) = z;  // z
        // Orientation (roll, pitch) remain unchanged
        next_state(6) = state(6); // roll
        next_state(7) = state(7); // pitch

        return next_state;
      }

      // Observation model
      static z_t observationModel(const x_t& state) {
        z_t observation;
        observation(0) = state(0); // x
        observation(1) = state(1); // y
        observation(2) = state(2); // z
        observation(3) = state(6); // roll
        observation(4) = state(7); // pitch
        observation(5) = state(8); // yaw
        return observation;
      }

      Eigen::VectorXd compute_visual_field(double heading, std::vector<x_t> all_states_, const VisualFieldParams& params, int agent_index) {
        Eigen::VectorXd visual_field = Eigen::VectorXd::Zero(params.field_size);
        for (int j = 0; j < uav_names.size(); j++) {
          if (j == agent_index) {
            continue;
          }
          const auto& state_j = all_states_[j];
          double xj = state_j(0) - odom_positions[agent_index](0);
          double yj = state_j(1) - odom_positions[agent_index](1);
          ROS_INFO_STREAM(odom_positions[agent_index]);
          
          double dij = sqrt(xj * xj + yj * yj);
          if (dij < 1e-6) {
            continue;
          }
          double phij = atan2(yj, xj);
          double delta_phij = atan(params.R / dij);
          double angle_uav_frame = atan2(sin(phij - heading), cos(phij - heading));
          int center_j = static_cast<int>((angle_uav_frame + M_PI) / (2 * M_PI / (params.field_size - 1))); // Center of the j UAV in the visual field
          int half_angle_width = static_cast<int>(delta_phij / (2 * M_PI / (params.field_size - 1)));
          for (unsigned int k = 0; k <= 2 * half_angle_width; k++) {
            if (dij > params.R) {
              int idx = (center_j - half_angle_width + k + params.field_size) % params.field_size;
              visual_field[idx] = 1;
            } else {
              // ROS_ERROR("[PoseFiltration]: uav_i above or under uav_i");
              continue;
            }
          }
        }

        return visual_field;
      }

      Eigen::VectorXd dPhi_V_of(const Eigen::VectorXd &Phi, const Eigen::VectorXd &V) {
        Eigen::VectorXd padV(V.size() + 2);
        padV << V(V.size() - 1), V, V(0);
        Eigen::VectorXd dPhi_V_raw = padV.tail(padV.size() - 1) - padV.head(padV.size() - 1);
        if (dPhi_V_raw(0) > 0 && dPhi_V_raw(dPhi_V_raw.size() - 1) > 0) {
          Eigen::VectorXd new_dPhi_V_raw = dPhi_V_raw.head(dPhi_V_raw.size() - 1);
          dPhi_V_raw = new_dPhi_V_raw;
        } else {
          Eigen::VectorXd new_dPhi_V_raw = dPhi_V_raw.tail(dPhi_V_raw.size() - 1);
          dPhi_V_raw = new_dPhi_V_raw;
        }
        return dPhi_V_raw;
      }

      std::pair<double, double> compute_state_variables(double vel_now, const Eigen::VectorXd &Phi, const Eigen::VectorXd &V_now, const VisualFieldParams& params) {
        Eigen::VectorXd dPhi_V = dPhi_V_of(Phi, V_now);
        Eigen::ArrayXd G = -V_now.array();
        Eigen::ArrayXd G_spike = dPhi_V.array().square();
        Eigen::ArrayXd sinPhi = Phi.array().sin();
        Eigen::ArrayXd cosPhi = Phi.array().cos();
        Eigen::ArrayXd integrand_dpsi = G * sinPhi;
        Eigen::ArrayXd integrand_dvel = G * cosPhi;
        double dphi = 2 * M_PI / params.field_size;
        double integral_dpsi = dphi * (0.5 * integrand_dpsi[0] + integrand_dpsi.segment(1, params.field_size - 2).sum() + 0.5 * integrand_dpsi[params.field_size - 1]);
        double integral_dvel = dphi * (0.5 * integrand_dvel[0] + integrand_dvel.segment(1, params.field_size - 2).sum() + 0.5 * integrand_dvel[params.field_size - 1]);
        double dpsi = params.BET0 * integral_dpsi + params.BET0 * params.BET1 * (G_spike * sinPhi).sum();
        double dvel = params.GAM * (params.V0 - vel_now) + params.ALP0 * integral_dvel + params.ALP0 * params.ALP1 * (G_spike * cosPhi).sum();
        // ROS_INFO("integral dvel %.9f ", integral_dvel);
        return std::make_pair(dvel, dpsi);
      }

      void loadParameters() {
        mrs_lib::ParamLoader param_loader(nh_, "PoseFiltration");
        param_loader.addYamlFileFromParam("config");
        param_loader.loadParam("uav_names", uav_names);
        param_loader.loadParam("fish_model_params/GAM", params.GAM);
        param_loader.loadParam("fish_model_params/ALP0", params.ALP0);
        param_loader.loadParam("fish_model_params/ALP1", params.ALP1);
        param_loader.loadParam("fish_model_params/ALP2", params.ALP2);
        param_loader.loadParam("fish_model_params/BET0", params.BET0);
        param_loader.loadParam("fish_model_params/BET1", params.BET1);
        param_loader.loadParam("fish_model_params/BET2", params.BET2);
        param_loader.loadParam("fish_model_params/V0", params.V0);
        param_loader.loadParam("fish_model_params/R", params.R);
        param_loader.loadParam("fish_model_params/field_size", params.field_size);
        param_loader.loadParam("filtration_params/q_vel", q_vel, 1.0);
        param_loader.loadParam("filtration_params/q_pos", q_pos, 0.1);
        param_loader.loadParam("filtration_params/q_ori", q_ori, 0.25);

        for (const auto& uav_name : uav_names) {
          double spawn_x, spawn_y, spawn_z, spawn_heading;
          std::string spawn_param_base = uav_name + "/spawn";
          if (!param_loader.loadParam(spawn_param_base + "/x", spawn_x) ||
            !param_loader.loadParam(spawn_param_base + "/y", spawn_y) ||
            !param_loader.loadParam(spawn_param_base + "/z", spawn_z) ||
            !param_loader.loadParam(spawn_param_base + "/heading", spawn_heading)) {
            ROS_ERROR_STREAM("[PoseFiltration]: Failed to load spawn parameters for " << uav_name);
            ros::shutdown();
          }
          ROS_INFO_STREAM("Loaded spawn parameters for " << uav_name << ": x=" << spawn_x << ", y=" << spawn_y << ", z=" << spawn_z << ", heading=" << spawn_heading);
          initial_positions.push_back({spawn_x, spawn_y, spawn_z, spawn_heading});
        }

        if (!param_loader.loadedSuccessfully()) {
          ROS_ERROR("[PoseFiltration]: Could not load all parameters!");
          ros::shutdown();
        }
      }

      void setupProcessNoise() {
        // Initialize process noise matrix (Q)
        process_noise_ = Q_t::Zero();

        // Position noise (x, y, z)
        process_noise_(0, 0) = q_pos;  // Variance in x position
        process_noise_(1, 1) = q_pos;  // Variance in y position
        process_noise_(2, 2) = q_pos;  // Variance in z position

        // Velocity noise (vx, vy, vz)
        process_noise_(3, 3) = q_vel;  // Variance in x velocity
        process_noise_(4, 4) = q_vel;  // Variance in y velocity
        process_noise_(5, 5) = q_vel;  // Variance in z velocity

        // Orientation noise (roll, pitch, yaw)
        process_noise_(6, 6) = q_ori;  // Variance in roll
        process_noise_(7, 7) = q_ori;  // Variance in pitch
        process_noise_(8, 8) = q_ori;  // Variance in yaw

        // Speed noise
        process_noise_(9, 9) = q_vel;  // Variance in speed
      }

      void initializeSubscribersAndPublishers() {
        for (const auto& uav_name : uav_names) {
          std::string topic_name = "/" + uav_name + "/uvdar/measuredPoses";
          ros::Subscriber sub = nh_.subscribe<mrs_msgs::PoseWithCovarianceArrayStamped>(
              topic_name, 1000, boost::bind(&PoseFiltration::uvdarCallback, this, _1, uav_name));
          uvdar_subscribers_.push_back(sub);

          std::string pub_topic_name = "/" + uav_name + "/uvdar/filteredPoses";
          ros::Publisher pub = nh_.advertise<mrs_msgs::PoseWithCovarianceArrayStamped>(pub_topic_name, 1);
          filtered_pose_publishers_.push_back(pub);

          std::string odom_topic_name = "/" + uav_name + "/estimation_manager/odom_main";
          ros::Subscriber odom_sub = nh_.subscribe<nav_msgs::Odometry>(
              odom_topic_name, 1000, boost::bind(&PoseFiltration::odomCallback, this, _1, uav_name));
          odom_subscribers.push_back(odom_sub);
        }
        std::string position_topic_name = "/multirotor_simulator/uav_poses";
        position_subscriber = nh_.subscribe<geometry_msgs::PoseArray>(
            position_topic_name, 10, &PoseFiltration::positionCallback, this);
        uav_positions.resize(uav_names.size());
        odom_positions.resize(uav_names.size(), Eigen::Vector3d::Zero());

      }

      void odomCallback(const nav_msgs::Odometry::ConstPtr& msg, const std::string& observer_name) {
        int observer_id = extractIdFromName(observer_name)-1;
        Eigen::Vector3d position;
        position(0) = msg->pose.pose.position.x;
        position(1) = msg->pose.pose.position.y;
        position(2) = msg->pose.pose.position.z;
        odom_positions[observer_id] = position;
      }

      void uvdarCallback(const mrs_msgs::PoseWithCovarianceArrayStamped::ConstPtr& msg, const std::string& observer_name) {
        int observer_id = extractIdFromName(observer_name)-1;
        // double dt = (ros::Time::now() - last_timestamp).toSec();
        timestamp = getCurrentTimeAsDouble();
        double dt = 0.1;
        // ROS_INFO_STREAM("[PoseFiltration]: dt "<< dt);
        last_timestamp = ros::Time::now();

        mrs_msgs::PoseWithCovarianceArrayStamped filtered_msg;
        filtered_msg.header = msg->header;

        // Store all received measurements in a map
        std::unordered_map<int, mrs_msgs::PoseWithCovarianceIdentified> measurements;
        for (const auto& pose : msg->poses) {
          // mrs_msgs::PoseWithCovarianceIdentified modified_pose = pose;
          // modified_pose.pose.position.x -= uav_positions[observer_id](0);    
          // modified_pose.pose.position.y -= uav_positions[observer_id](1); 
          // modified_pose.pose.position.z -= uav_positions[observer_id](2);       
          measurements[pose.id-1] = pose;
        }
        all_states_.clear();
        all_states_.resize(uav_names.size(), x_t::Zero()); // Ensure it has the correct size
        for (size_t target_id = 0; target_id < uav_names.size(); target_id++) {
          // Set the current states in all_states_
          if (ukf_map_[observer_id].find(target_id) != ukf_map_[observer_id].end()) {
            all_states_[target_id] = ukf_map_[observer_id][target_id].x; // Populate the state vector
          } else {
              all_states_[target_id] = x_t::Zero(); // Set to zero if the target state is not initialized
          }
        }
        
        // std::vector<x_t> all_states_backup = all_states_;
        // Loop through all possible target IDs
        for (size_t target_id = 0; target_id < uav_names.size(); target_id++) {
          // if (target_id = observer_id) {
          //   continue;
          // }
          // if (filtration_allowed_) {  
          //   auto transformed_states = transformAllStatesToLocalFrame(target_id, all_states_);
          //   all_states_.clear();
          //   all_states_.resize(uav_names.size(), x_t::Zero());
          //   all_states_ = transformed_states;
          // }
          // If a measurement exists, predict and correct
          
          current_agent_id_ = target_id;
          if (measurements.find(target_id) != measurements.end() && filtration_allowed_) {
            const auto& pose = measurements[target_id];
            logMeasured(observer_id, target_id, pose.pose, timestamp);
            ukf_map_[observer_id][target_id] = predictAndCorrect(observer_id, target_id, pose, dt);
          } else {
            // No measurement: only predict
            initializeUKF(observer_id, target_id, initial_positions);
            if (filtration_allowed_) {  
              try {
                ukf_map_[observer_id][target_id] = ukf_.predict(ukf_map_[observer_id][target_id], ukf_t::u_t::Zero(), process_noise_, dt);
              } catch (const std::exception& e) {
                ROS_ERROR("[PoseFiltration]: UKF correction failed for observer %d, target %ld: %s", observer_id, target_id, e.what());
              }  
            }
          }
          
          if (filtration_allowed_) {  
            logPredicted(observer_id, target_id, ukf_map_[observer_id][target_id].x, timestamp);
            logGroundTruth(target_id, odom_positions[target_id], timestamp);
          }
          
          
          // Add the filtered state to the outgoing message
          mrs_msgs::PoseWithCovarianceIdentified filtered_pose = getFilteredPose(observer_id, target_id);
          filtered_msg.poses.push_back(filtered_pose);
          // all_states_.clear();
          // all_states_.resize(uav_names.size(), x_t::Zero());
          // all_states_ = all_states_backup;
        }
        // ROS_INFO_STREAM("Current UKF state covariance of uav 1 is:" << std::endl << ukf_map_[observer_id][0].P);


        // Publish the filtered message
        // auto observer_index = std::distance(uav_names.begin(), std::find(uav_names.begin(), uav_names.end(), observer_name));
        // if (observer_index < filtered_pose_publishers_.size()) {
        //   filtered_pose_publishers_[observer_index].publish(filtered_msg);
        // }
        filtered_pose_publishers_[observer_id].publish(filtered_msg);
      }

      void positionCallback(const geometry_msgs::PoseArray::ConstPtr& msg){
        // std::string frame_id = msg->header.frame_id;
        // std::string uav_name = extractUavNameFromFrameId(frame_id);
        // int uav_id = extractIdFromName(uav_name);
        
        double now = getCurrentTimeAsDouble();
        last_callback_time_ = now;
        int id = 0;
        for (const auto& pose : msg->poses) {
          Eigen::Vector3d pos;
          pos(0) = pose.position.x;
          pos(1) = pose.position.y;
          pos(2) = pose.position.z;
          uav_positions[id] = (pos);
          id++;
        } 
      }

      statecov_t predictAndCorrect(int observer_id, int target_id, const mrs_msgs::PoseWithCovarianceIdentified& pose, double dt) {
        initializeUKF(observer_id, target_id, initial_positions);

        z_t measurement;
        measurement(0) = pose.pose.position.x;
        measurement(1) = pose.pose.position.y;
        measurement(2) = pose.pose.position.z;
        Eigen::Quaterniond q(
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z
        );
        q.normalize();
        Eigen::Vector3d rpy = quaternionToRPY(q.x(), q.y(), q.z(), q.w());
        measurement(3) = rpy(0);
        measurement(4) = rpy(1);
        measurement(5) = rpy(2);

        R_t measurement_covariance;
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j) {
            measurement_covariance(i, j) = pose.covariance[i * 6 + j];
          }
        }
        // ROS_INFO_STREAM("Measurement covariance matrix:\n" << measurement_covariance);

        auto& state = ukf_map_[observer_id][target_id];

        try {
          state = ukf_.predict(state, ukf_t::u_t::Zero(), process_noise_, dt);
          state = ukf_.correct(state, measurement, measurement_covariance);
        } catch (const std::exception& e) {
          ROS_ERROR("[PoseFiltration]: UKF correction failed for observer %d, target %d: %s", observer_id, target_id, e.what());
        }

        return state;
      }

      mrs_msgs::PoseWithCovarianceIdentified getFilteredPose(int observer_id, int target_id) {
        const auto& state_cov = ukf_map_[observer_id][target_id];
        const auto& state = state_cov.x;

        mrs_msgs::PoseWithCovarianceIdentified filtered_pose;
        filtered_pose.id = target_id+1;
        filtered_pose.pose.position.x = state(0);
        filtered_pose.pose.position.y = state(1);
        filtered_pose.pose.position.z = state(2);
        return filtered_pose;
      }

      void initializeUKF(int observer_id, int target_id, const std::vector<Position>& initial_positions) {
        if (ukf_map_[observer_id].find(target_id) == ukf_map_[observer_id].end()) {
          x_t x0 = x_t::Zero();
          ukf_t::P_t P0 = ukf_t::P_t::Identity();  
          P0(0, 0) = 0.01;
          P0(1, 1) = 0.01;
          P0(2, 2) = 0.01;

          if (target_id < initial_positions.size()) {
            // x0(0) = initial_positions[target_id].x - initial_positions[observer_id].x;
            // x0(1) = initial_positions[target_id].y - initial_positions[observer_id].y;
            // x0(2) = initial_positions[target_id].z - initial_positions[observer_id].z;
            // x0(8) = initial_positions[target_id].heading - initial_positions[observer_id].heading;
            x0(0) = initial_positions[target_id].x;
            x0(1) = initial_positions[target_id].y;
            x0(2) = initial_positions[target_id].z;
            x0(8) = initial_positions[target_id].heading;
          }
          ukf_map_[observer_id][target_id] = {x0, P0};
        }
      }

      Eigen::Vector3d quaternionToRPY(double qx, double qy, double qz, double qw) {
        Eigen::Vector3d rpy;
        double sinr_cosp = 2.0 * (qw * qx + qy * qz);
        double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
        rpy(0) = std::atan2(sinr_cosp, cosr_cosp);

        double sinp = 2.0 * (qw * qy - qz * qx);
        rpy(1) = std::abs(sinp) >= 1 ? std::copysign(M_PI / 2, sinp) : std::asin(sinp);

        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        rpy(2) = std::atan2(siny_cosp, cosy_cosp);

        return rpy;
      }

      int extractIdFromName(const std::string& uav_name) {
        return std::stoi(uav_name.substr(3));
      }

      bool activationServiceCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        // service for activation of planning
        ROS_INFO("[FishModelSimulator]: Activation service called.");
        res.success = true;

        if (filtration_allowed_) {
          res.message = "Control was already allowed.";
          ROS_WARN("[AreaMonitoringController]: %s", res.message.c_str());
        } else {
          filtration_allowed_ = true;
        }


        return true;
      }

      Eigen::Matrix3d computeRotationMatrix(double roll, double pitch, double yaw) {
        Eigen::Matrix3d R_x, R_y, R_z;
        R_x << 1, 0, 0,
              0, cos(roll), -sin(roll),
              0, sin(roll), cos(roll);
        R_y << cos(pitch), 0, sin(pitch),
              0, 1, 0,
              -sin(pitch), 0, cos(pitch);
        R_z << cos(yaw), -sin(yaw), 0,
              sin(yaw), cos(yaw), 0,
              0, 0, 1;
        return R_z * R_y * R_x;
    }

    std::vector<x_t> transformAllStatesToLocalFrame(int target_id, std::vector<x_t> obsv_states) {
      // Target UAV's state in the observer frame
      const x_t& target_state = obsv_states[target_id];

      // Compute the rotation matrix for the target UAV's local frame
      Eigen::Matrix3d R_target = computeRotationMatrix(target_state(6), target_state(7), target_state(8));
      Eigen::Matrix3d R_target_transpose = R_target.transpose();

      // Transform all states into the target UAV's local frame
      for (int i = 0; i < obsv_states.size(); i++) {
        x_t& state = obsv_states[i]; // Reference to the current UAV's state

        if (i == target_id) {
          // Transform target UAV's velocity and orientation into its own frame
          Eigen::Vector3d velocity_observer_frame(state(3), state(4), state(5));
          Eigen::Vector3d velocity_target_frame = R_target_transpose * velocity_observer_frame;
          // Update the target UAV's state
          state(0) = 0.0; state(1) = 0.0; state(2) = 0.0; // Position is zero in its own frame
          state(3) = velocity_target_frame(0);
          state(4) = velocity_target_frame(1);
          state(5) = velocity_target_frame(2);
          state(6) = 0;
          state(7) = 0;
          state(8) = 0;
          continue;
        }

        // Transform position (requires translation and rotation)
        Eigen::Vector3d position_observer_frame(state(0), state(1), state(2));
        Eigen::Vector3d position_target_frame = R_target_transpose * (position_observer_frame - target_state.head<3>());

        // Transform velocity (requires rotation only)
        // Eigen::Vector3d velocity_observer_frame(state(3), state(4), state(5));
        // Eigen::Vector3d velocity_target_frame = R_target_transpose * velocity_observer_frame;

        // // Transform orientation
        // Eigen::Matrix3d R_orientation_observer = computeRotationMatrix(state(6), state(7), state(8));
        // Eigen::Matrix3d R_orientation_target = R_target_transpose * R_orientation_observer;
        // Eigen::Vector3d orientation_target_frame = computeEulerAnglesFromRotationMatrix(R_orientation_target);

        // Update the transformed state directly
        state(0) = position_target_frame(0);
        state(1) = position_target_frame(1);
        state(2) = position_target_frame(2);
        // state(3) = velocity_target_frame(0);
        // state(4) = velocity_target_frame(1);
        // state(5) = velocity_target_frame(2);
        // state(6) = orientation_target_frame(0);
        // state(7) = orientation_target_frame(1);
        // state(8) = orientation_target_frame(2);
      }

      // ROS_INFO("[PoseFiltration]: Transformed all_states_ positions to target_id %d local frame", target_id);
      return obsv_states;
    }
    Eigen::Vector3d computeEulerAnglesFromRotationMatrix(const Eigen::Matrix3d& rotation_matrix) {
      Eigen::Vector3d euler_angles;

      // Extract roll (x-axis rotation)
      euler_angles(0) = std::atan2(rotation_matrix(2, 1), rotation_matrix(2, 2));

      // Extract pitch (y-axis rotation)
      double sin_pitch = -rotation_matrix(2, 0);
      if (std::abs(sin_pitch) >= 1) {
        euler_angles(1) = std::copysign(M_PI / 2, sin_pitch); // Use 90 degrees if out of range
      } else {
        euler_angles(1) = std::asin(sin_pitch);
      }

      // Extract yaw (z-axis rotation)
      euler_angles(2) = std::atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));

      return euler_angles;
    }

    void initializeFiles() {
      std::string package_path = ros::package::getPath("fish_model_simulator");
      predicted_file_.open(package_path + "/predicted_positions.csv");
      measured_file_.open(package_path + "/measured_positions.csv");
      groundtruth_file_.open(package_path + "/groundtruth_positions.csv");
      if (predicted_file_.is_open() && measured_file_.is_open() && groundtruth_file_.is_open()) {
        files_initialized_ = true;
        predicted_file_ << "timestamp,obsv_id,target_id,x,y,z\n";
        measured_file_ << "timestamp,obsv_id,target_id,x,y,z\n";
        groundtruth_file_ << "timestamp,id,x,y,z\n";
      } else {
        ROS_ERROR("[PoseFiltration]: Failed to open output files for logging");
      }
    }

    void closeFiles() {
      if (predicted_file_.is_open()) {
        predicted_file_.close();
      }
      if (measured_file_.is_open()) {
        measured_file_.close();
      }
      if (groundtruth_file_.is_open()) {
        groundtruth_file_.close();
      }
      ROS_INFO("[PoseFiltration]: Output files successfully closed.");
    }

    void logPredicted(int observer_id, int target_id, const x_t& state, double timestamp) {
      if (files_initialized_) {
        predicted_file_ << std::fixed << std::setprecision(9) << timestamp << "," << observer_id << "," << target_id << ","
                        << state(0) << "," << state(1) << "," << state(2) << "\n";
      }
    }

    void logMeasured(int observer_id, int target_id, const geometry_msgs::Pose& pose, double timestamp) {
      if (files_initialized_) {
        measured_file_ << std::fixed << std::setprecision(9) << timestamp << "," << observer_id << "," << target_id << ","
                        << pose.position.x << "," << pose.position.y << "," << pose.position.z << "\n";
      }
    }

    void logGroundTruth(int id, const Eigen::Vector3d& position, double timestamp) {
      if (files_initialized_) {
        groundtruth_file_ << std::fixed << std::setprecision(9) << timestamp << "," << id << ","
                          << position(0) << "," << position(1) << "," << position(2) << "\n";
      }
    }
    
    double getCurrentTimeAsDouble() {
      ros::Time current_time = ros::Time::now();
      return current_time.sec + current_time.nsec * 1e-9;
    }
    
  };

} // namespace filtration

int main(int argc, char** argv) {
  ros::init(argc, argv, "pose_filtration");

  ros::NodeHandle nh("~");
  filtration::PoseFiltration pf(nh);
  ROS_INFO("[PoseFiltration]: Pose Filtration node initiated");
  ros::spin();
  return 0;
}