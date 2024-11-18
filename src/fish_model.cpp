/* includes //{ */

#include <ros/ros.h>


#include <rosgraph_msgs/Clock.h>

#include <geometry_msgs/PoseArray.h>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/publisher_handler.h>

#include <dynamic_reconfigure/server.h>

#include <Eigen/Dense>

#include <mrs_msgs/VelocityReferenceStampedSrv.h>
#include <mrs_msgs/ReferenceStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/TransformStamped.h>
#include <mrs_lib/transformer.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/service_client_handler.h>

//}

#include <unordered_map>
#include <mrs_msgs/HwApiVelocityHdgRateCmd.h>
#include <std_srvs/Trigger.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <mrs_msgs/PoseWithCovarianceArrayStamped.h>  
#include <mrs_msgs/PoseWithCovarianceIdentified.h>
#include <mutex>

namespace fish_model {

	class FishModelSimulator{

	public:
		FishModelSimulator(ros::NodeHandle& nh);
    void run();

	private:
		ros::NodeHandle   nh_;
		std::atomic<bool> is_initialized_;
		// | ------------------------- params ------------------------- |

		double  _simulation_rate_ = 100.0;

		// | -------------- custom added -------------------------------|
		std::mutex transformer_mutex;
		mrs_lib::Transformer transformer_;

		mrs_msgs::VelocityReferenceStampedSrv vel_srv_;
		
		std::vector<ros::ServiceClient> client_vel_ref_arr;
		std::vector<ros::Subscriber> uvdar_subscribers_;
		std::vector<ros::Subscriber> velocity_subscribers;
		void updateVelocities(void);

		std::unordered_map<int, std::unordered_map<int, mrs_msgs::PoseWithCovarianceIdentified>> uav_poses_map;
    std::mutex uav_poses_mutex;
    std::condition_variable uav_poses_cv;
		std::vector<geometry_msgs::Vector3Stamped> uav_velocity_msgs;
    std::mutex uav_velocity_mutex;


		Eigen::VectorXd dPhi_V_of(const Eigen::VectorXd &Phi, const Eigen::VectorXd &V);
		std::pair<double, double> compute_state_variables(double vel_now, const Eigen::VectorXd &Phi, const Eigen::VectorXd &V_now);
		Eigen::VectorXd compute_visual_field(double heading, int agent_index, int N);
		bool activationServiceCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
		double wrapAngle(double angle);
		void uvdarCallback(const mrs_msgs::PoseWithCovarianceArrayStamped::ConstPtr& msg, const std::string& observer_name);
		int extractIdFromName(const std::string& uav_name);
		void velocityCallback(const geometry_msgs::Vector3Stamped::ConstPtr& msg);
		std::string extractUavNameFromFrameId(const std::string& frame_id);
    void waitForNonEmptyPoses();
    

		ros::ServiceServer service_activate_control_;

		int computeLargestSubgroup(double distance_threshold);
		double computeOverlap(double overlap_threshold);
		double computeMeanDistance();
		double computePolarization();
		void writeMetricsToCSV(double time, double polarization, double mean_distance, double overlap, int largest_subgroup);

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
		bool control_allowed_ = false;
		double xmin = -20, xmax = 20, ymin = -20, ymax = 20;
		std::string output_frame_;
		std::vector<std::string> uav_names;
	};

	FishModelSimulator::FishModelSimulator(ros::NodeHandle& nh) : nh_(nh) {
		is_initialized_ = false;

		mrs_lib::ParamLoader param_loader(nh_, "FishModelSimulator");
		param_loader.addYamlFileFromParam("config");
		param_loader.loadParam("fish_model_params/GAM", GAM);
		param_loader.loadParam("fish_model_params/ALP0", ALP0);
		param_loader.loadParam("fish_model_params/ALP1", ALP1);
		param_loader.loadParam("fish_model_params/ALP2", ALP2);
		param_loader.loadParam("fish_model_params/BET0", BET0);
		param_loader.loadParam("fish_model_params/BET1", BET1);
		param_loader.loadParam("fish_model_params/BET2", BET2);
		param_loader.loadParam("fish_model_params/V0", V0);
		param_loader.loadParam("fish_model_params/R", R);
		param_loader.loadParam("fish_model_params/field_size", field_size, 16384);
		param_loader.loadParam("output_frame", output_frame_);

		double clock_rate;
		param_loader.loadParam("clock_rate", clock_rate);

		param_loader.loadParam("uav_names", uav_names);
    ROS_INFO("uav names loaded");

		// mrs_lib::SubscribeHandlerOptions shopts;
		// shopts.nh = nh_;
		// shopts.node_name = "FishModelSimulator";
		// shopts.no_message_timeout = mrs_lib::no_timeout;
		// shopts.threadsafe = true;
		// shopts.autostart = true;
		// shopts.queue_size = 10;
		// shopts.transport_hints = ros::TransportHints().tcpNoDelay();
		service_activate_control_ = nh_.advertiseService("control_activation_in", &FishModelSimulator::activationServiceCallback, this);

		for (size_t i = 0; i < uav_names.size(); i++) {
			std::string uav_name = uav_names.at(i);
			ros::ServiceClient client_vel_ref_ = nh_.serviceClient<mrs_msgs::VelocityReferenceStampedSrv>("/" + uav_name + "/control_manager/velocity_reference");
			client_vel_ref_arr.push_back(client_vel_ref_);
		}
		for (size_t i = 0; i < uav_names.size(); i++) {
			std::string uav_name = uav_names.at(i);

			std::string uvdar_topic_name = "/" + uav_name + "/uvdar/filteredPoses";
			ros::Subscriber uvdar_sub = nh_.subscribe<mrs_msgs::PoseWithCovarianceArrayStamped>(
				uvdar_topic_name, 10, boost::bind(&FishModelSimulator::uvdarCallback, this, _1, uav_name));
			uvdar_subscribers_.push_back(uvdar_sub);

			std::string filter_topic_name = "/" + uav_name + "/hw_api/velocity";
			ros::Subscriber filter_sub = nh_.subscribe<geometry_msgs::Vector3Stamped>(
				filter_topic_name, 10, &FishModelSimulator::velocityCallback, this);
			velocity_subscribers.push_back(filter_sub);
		}
    uav_velocity_msgs.resize(uav_names.size());
    transformer_ = mrs_lib::Transformer("FishModelTransformer");
		is_initialized_ = true;
    		
	}

	//}

	// | ------------------------- timers ------------------------- |

	/* timerMain() //{ */

  void FishModelSimulator::run() {
    if (!is_initialized_) {
      ROS_ERROR("[FishModelSimulator]: Initialization failed. Exiting run loop.");
      return;
    }

    ROS_INFO("[FishModelSimulator]: Entering run loop...");

    // Start an AsyncSpinner with 1 thread to process callbacks in the background
    ros::AsyncSpinner spinner(1);
    spinner.start();  // This will allow callbacks (e.g., uvdarCallback) to run concurrently

    // Wait for the initial condition before starting the loop
    waitForNonEmptyPoses();

    // Set the loop rate (e.g., 10 Hz)
    ros::Rate rate(10); // Adjust frequency as needed
    while (ros::ok()) {
      ROS_INFO_ONCE("[FishModelSimulator]: main loop running");

      updateVelocities();

      rate.sleep();  // Sleep to maintain the loop rate
    }

    ROS_INFO("[FishModelSimulator]: Exiting run loop...");
    spinner.stop();  // Stop the spinner when the node is shutting down
  }


	void FishModelSimulator::updateVelocities(void) {
		// Update the UAVs' odometry vector
		if (!control_allowed_) {
			ROS_WARN_THROTTLE(3.0, "[AreaMonitoringController]: Waiting for activation.");
			return;
		}
		// Define variables
		Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(field_size, -M_PI, M_PI);
		bool send_message = false;
		
    std::lock_guard<std::mutex> lock(uav_velocity_mutex);
		// For each UAV, update velocities
		for (size_t i = 0; i < uav_names.size(); i++) {
			// Retrieve current velocities and compute current heading and speed
			geometry_msgs::Vector3Stamped uav_velocity = uav_velocity_msgs[i];  // velocity msg
			auto input2output_tmp = transformer_.getTransform(uav_velocity.header.frame_id, output_frame_, uav_velocity.header.stamp-ros::Duration(0.2)); 
			if (!input2output_tmp){
				ROS_ERROR_STREAM_THROTTLE(1.0,"[UVDARMultirobotSimulator]: Could not obtain transform from " << uav_velocity.header.frame_id<< " to " <<  output_frame_ << "!");
				return;
			}
			geometry_msgs::TransformStamped input2output = input2output_tmp.value();
			auto uav_velocity_global_tmp = transformer_.transform(uav_velocity, input2output);
			if (!uav_velocity_global_tmp){
				ROS_ERROR_STREAM_THROTTLE(1.0,"[UVDARMultirobotSimulator]: Could not perform the transform from " << uav_velocity.header.frame_id<< " to " <<  output_frame_ << "!");
				return;
			}
			auto uav_velocity_global = uav_velocity_global_tmp.value();
			float vx_global = uav_velocity_global.vector.x;
			float vy_global = uav_velocity_global.vector.y;
			double velocity = sqrt(pow(vx_global, 2) + pow(vy_global, 2)); //norm of velocity
			double heading = atan2(vy_global, vx_global); //global coord heading

			// Compute the visual field for the current UAV

			Eigen::VectorXd visual_field = FishModelSimulator::compute_visual_field(heading, i, uav_names.size());
      int vis_field_sum = visual_field.sum();
      // ROS_INFO_STREAM("vis field sum:  "<< vis_field_sum);
			// Compute state variables dvel and dpsi
			auto [dvel, dpsi] = FishModelSimulator::compute_state_variables(velocity, phi, visual_field);
			// Update speed along the heading direction by integrating dvel over the time step
			velocity += dvel ; //* _clock_min_dt_;

			heading += dpsi; 
			// Compute new velocity components based on updated speed and heading
			double vel_x_new = velocity * cos(heading);
			double vel_y_new = velocity * sin(heading);

			// Update the service request velocities
      std::string vel_srv_output_frame = "world_origin";
			vel_srv_.request.reference.reference.velocity.x = vel_x_new;
			vel_srv_.request.reference.reference.velocity.y = vel_y_new;
			vel_srv_.request.reference.reference.velocity.z = 0.0;
			vel_srv_.request.reference.reference.use_heading = true;
			vel_srv_.request.reference.reference.heading = heading; 
      vel_srv_.request.reference.header.frame_id = vel_srv_output_frame;


			// ROS_INFO("calculated variables state: dpsi = %.9f", dpsi); 
			// Send velocity command if the UAV is above the minimum height
			if (client_vel_ref_arr[i].call(vel_srv_)) {
			if (send_message) {
				ROS_INFO("Service call successful for UAV %zu", i);
			}
			} else {
				ROS_ERROR("Service call unsuccessful for UAV %zu", i);
			}
		}
		// ROS_INFO("Velocity sent to uav x: %.3f y: %.3f", vel_srv_.request.reference.reference.velocity.x, vel_srv_.request.reference.reference.velocity.y);
	}


	Eigen::VectorXd FishModelSimulator::dPhi_V_of(const Eigen::VectorXd &Phi, const Eigen::VectorXd &V) {

    Eigen::VectorXd padV(V.size() + 2);

		padV << V(V.size() - 1), V, V(0);

		Eigen::VectorXd dPhi_V_raw = padV.tail(padV.size() - 1) - padV.head(padV.size() - 1);

		if (dPhi_V_raw(0) > 0 && dPhi_V_raw(dPhi_V_raw.size() - 1) > 0) {
      dPhi_V_raw.conservativeResize(dPhi_V_raw.size() - 1);

		} else {
			Eigen::VectorXd new_dPhi_V_raw = dPhi_V_raw.segment(1, dPhi_V_raw.size() - 1);
      dPhi_V_raw = new_dPhi_V_raw;

		}

		return dPhi_V_raw;
	}

	std::pair<double, double> FishModelSimulator::compute_state_variables(double vel_now, const Eigen::VectorXd &Phi, const Eigen::VectorXd &V_now) {
		// Compute dPhi_V

		Eigen::VectorXd dPhi_V = dPhi_V_of(Phi, V_now);

		// G is -V_now
		Eigen::ArrayXd G = -V_now.array();

		// G_spike is the square of dPhi_V
		Eigen::ArrayXd G_spike = dPhi_V.array().square();

		// Precompute sine and cosine of Phi  
		Eigen::ArrayXd sinPhi = Phi.array().sin();
		Eigen::ArrayXd cosPhi = Phi.array().cos();

		// Compute the integrand values
		Eigen::ArrayXd integrand_dpsi = G * sinPhi;
		Eigen::ArrayXd integrand_dvel = G * cosPhi;

		// Apply the trapezoidal rule using vectorized operations
		double dphi = 2 * M_PI / field_size;

		double integral_dpsi = dphi * (0.5 * integrand_dpsi[0] + integrand_dpsi.segment(1, field_size - 2).sum() + 0.5 * integrand_dpsi[field_size - 1]);

		// Similarly for integral_dvel
		double integral_dvel = dphi * (0.5 * integrand_dvel[0] + integrand_dvel.segment(1, field_size - 2).sum() + 0.5 * integrand_dvel[field_size - 1]);

		double dpsi = BET0 * integral_dpsi + BET0 * BET1 * (G_spike * sinPhi).sum();
		double dvel = GAM * (V0 - vel_now) + ALP0 * integral_dvel + ALP0 * ALP1 * (G_spike * cosPhi).sum();

		// ROS_INFO("integral dvel %.9f ", integral_dvel);

		return std::make_pair(dvel, dpsi);
	}

	Eigen::VectorXd FishModelSimulator::compute_visual_field(double heading, int agent_index, int N) {
		Eigen::VectorXd visual_field = Eigen::VectorXd::Zero(field_size);
    std::lock_guard<std::mutex> lock(uav_poses_mutex);
		for (int j = 0; j < N; j++) {
			if (agent_index != j) {
        // Extract x and y coordinates from the position vectors
        if (uav_poses_map.find(agent_index) != uav_poses_map.end() && 
            uav_poses_map[agent_index].find(j) != uav_poses_map[agent_index].end()) {
          mrs_msgs::PoseWithCovarianceIdentified pose_j = uav_poses_map[agent_index][j];
          double xj = pose_j.pose.position.x; 
          double yj = pose_j.pose.position.y;

            // Calculate the distance between agent i and agent j
          double dij = sqrt(xj * xj + yj * yj);

          // Calculate the relative angle between the agents
          double phij = atan2(yj, xj);
          // ROS_INFO_STREAM("agent index:  " << agent_index << "j:  " << j << "dij:  " << dij << "phij:  " << phij);
          // Calculate the angular width of the agent in the visual field
          double delta_phij = atan(R / dij);

          double angle_uav_frame = atan2( sin(phij - heading), cos(phij - heading) );
          int center_j = (angle_uav_frame + M_PI) / (2 * M_PI / (field_size - 1)); //center of the j uav in the list of the visual field in the i uav frame of reference to psi
          int half_angle_width = static_cast<int>(delta_phij / (2 * M_PI / (field_size - 1)));

          for (unsigned int k = 0; k <= 2 * half_angle_width; k++) {
            if(dij>R){
              int idx = (center_j - half_angle_width + k + field_size) % field_size;
              visual_field[idx] = 1;
            }else{
              ROS_ERROR("uav_i above or under uav_i");
            }
          }
        }else{ // need to handle this properly, this is just a temporary solution
          continue;
        }
        
			
			}
		}

		return visual_field;
	}


	bool FishModelSimulator::activationServiceCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
		// service for activation of planning
		ROS_INFO("[FishModelSimulator]: Activation service called.");
		res.success = true;

		if (control_allowed_) {
			res.message = "Control was already allowed.";
			ROS_WARN("[AreaMonitoringController]: %s", res.message.c_str());
		} else {
			control_allowed_ = true;
		}


		return true;
	}


	void FishModelSimulator::uvdarCallback(const mrs_msgs::PoseWithCovarianceArrayStamped::ConstPtr& msg, const std::string& observer_name) {
		int observer_id = extractIdFromName(observer_name);  // Assume a function to extract ID from the UAV name

		// ROS_INFO("Received UVDAR poses for observer %s (ID %d)", observer_name.c_str(), observer_id);
    std::lock_guard<std::mutex> lock(uav_poses_mutex);
		for (const auto& pose : msg->poses) {
			int target_id = pose.id;
			// ROS_INFO("Storing pose for observer ID %d and target ID %d:", observer_id, target_id);
			
			// Store the pose in the nested map
			uav_poses_map[observer_id-1][target_id-1] = pose;
		}
    uav_poses_cv.notify_all();
	}

	void FishModelSimulator::velocityCallback(const geometry_msgs::Vector3Stamped::ConstPtr& msg){
		std::string frame_id = msg->header.frame_id;
    std::string uav_name = extractUavNameFromFrameId(frame_id);
		int uav_id = extractIdFromName(uav_name);
    std::lock_guard<std::mutex> lock(uav_velocity_mutex);
    uav_velocity_msgs[uav_id-1] = *msg;
	}

	int FishModelSimulator::extractIdFromName(const std::string& uav_name) {
		// Assuming UAV names are formatted as "uavX", where X is the ID
		return std::stoi(uav_name.substr(3));
	}
	std::string FishModelSimulator::extractUavNameFromFrameId(const std::string& frame_id) {
	// Extract and return the substring up to the '/' delimiter
		return frame_id.substr(0, frame_id.find('/'));
	}

  void FishModelSimulator::waitForNonEmptyPoses() {
    std::unique_lock<std::mutex> lock(uav_poses_mutex);
    // ROS_INFO_STREAM("UAV POSES MAP SIZE:  " << uav_poses_map.size() << "UAV NAMES SIZE:  " << uav_names.size());
    uav_poses_cv.wait(lock, [this]() {
        // Check if the size of the map matches the expected number of observers
        if (uav_poses_map.size() != uav_names.size()) {
            return false;  // Not all observers are present in the map
        }

        // Iterate through the map to check if each observer has at least one target ID entry
        for (const auto& observer_entry : uav_poses_map) {
            const auto& target_map = observer_entry.second;

            // Check if the target map for the observer is empty
            if (target_map.empty()) {
                return false;  // Condition not met if any observer's target map is empty
            }
        }

        // If all conditions are met, return true to continue execution
        return true;
    });

    // Execution continues once the condition is satisfied
    ROS_INFO("All observers are present, and each has at least one target ID entry in uav_poses_map.");
}

}  // namespace fish_model

int main(int argc, char** argv) {
  ros::init(argc, argv, "fish_model_simulator");
  ros::NodeHandle nh("~");
  fish_model::FishModelSimulator fishModelSimulator(nh);
  fishModelSimulator.run();
  return 0;
}
