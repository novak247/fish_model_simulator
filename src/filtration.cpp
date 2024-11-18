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

namespace filtration {

  class PoseFiltration {
    private:
      ros::NodeHandle nh_;
      std::vector<ros::Subscriber> uvdar_subscribers_;
      std::vector<ros::Publisher> filtered_pose_publishers_;
      std::vector<std::string> uav_names;

      // Nested unordered_map to store poses for each observer and target
      std::unordered_map<int, std::unordered_map<int, mrs_msgs::PoseWithCovarianceIdentified>> uav_poses_map_;
      std::unordered_map<int, std::unordered_map<int, std::deque<mrs_msgs::PoseWithCovarianceIdentified>>> pose_history_map_;
      int history_size_; // Number of past measurements to store for averaging

      // Parameters for the filter
      double process_noise_;
      double measurement_noise_;

    public:
      PoseFiltration(ros::NodeHandle& nh) : nh_(nh) {
        loadParameters();
        initializeSubscribersAndPublishers();
      }

      void loadParameters() {
        mrs_lib::ParamLoader param_loader(nh_, "PoseFiltration");
        param_loader.addYamlFileFromParam("config");
        param_loader.loadParam("uav_names", uav_names);
        param_loader.loadParam("process_noise", process_noise_, 0.01);
        param_loader.loadParam("measurement_noise", measurement_noise_, 0.1);
        param_loader.loadParam("history_size", history_size_, 5);

        if (!param_loader.loadedSuccessfully()) {
          ROS_ERROR("[PoseFiltration]: Could not load all parameters!");
          ros::shutdown();
        }
      }

      void initializeSubscribersAndPublishers() {
        for (const auto& uav_name : uav_names) {
          // Initialize subscribers
          std::string topic_name = "/" + uav_name + "/uvdar/measuredPoses";
          ros::Subscriber sub = nh_.subscribe<mrs_msgs::PoseWithCovarianceArrayStamped>(
              topic_name, 1000, boost::bind(&PoseFiltration::uvdarCallback, this, _1, uav_name));
          uvdar_subscribers_.push_back(sub);
          ROS_INFO_STREAM("[PoseFiltration]: Subscribed to topic: " << topic_name);

          // Initialize publishers
          std::string pub_topic_name = "/" + uav_name + "/uvdar/filteredPoses";
          ros::Publisher pub = nh_.advertise<mrs_msgs::PoseWithCovarianceArrayStamped>(pub_topic_name, 1);
          filtered_pose_publishers_.push_back(pub);
          ROS_INFO_STREAM("[PoseFiltration]: Advertising filtered poses on topic: " << pub_topic_name);
        }
      }

      void uvdarCallback(const mrs_msgs::PoseWithCovarianceArrayStamped::ConstPtr& msg, const std::string& observer_name) {
        int observer_id = extractIdFromName(observer_name);

        mrs_msgs::PoseWithCovarianceArrayStamped filtered_msg;
        filtered_msg.header = msg->header;
        int last_target_id = 0;
        for (const auto& pose : msg->poses) {
          int target_id = pose.id;
          // ROS_INFO("Storing and filtering pose for observer ID %d and target ID %d", observer_id, target_id);
          if (last_target_id != 0 && target_id > last_target_id + 1) {
            // Fill in missing target_id poses with the last known position
            for (int missing_id = last_target_id + 1; missing_id < target_id; missing_id++) {
              if (uav_poses_map_[observer_id].find(missing_id) != uav_poses_map_[observer_id].end() && missing_id != observer_id) {
                mrs_msgs::PoseWithCovarianceIdentified filtered_pose = filterPose(uav_poses_map_[observer_id][missing_id], observer_id);
                filtered_msg.poses.push_back(filtered_pose);
              }
              else if (missing_id != observer_id){
                ROS_INFO_ONCE("Drone with missing id %d has not yet been seen by observer %d", missing_id, observer_id);
              }
              else{
                continue; // observer id = missing id, nothing should happen by design
              }
            }
          }
          // Store the raw pose
          uav_poses_map_[observer_id][target_id] = pose;

          // Filter the pose
          mrs_msgs::PoseWithCovarianceIdentified filtered_pose = filterPose(pose, observer_id);

          // // Add the filtered pose to the outgoing message
          filtered_msg.poses.push_back(filtered_pose);
          last_target_id = target_id;
        }

        // Find the index of the observer's publisher and publish the filtered message
        auto observer_index = std::distance(uav_names.begin(), std::find(uav_names.begin(), uav_names.end(), observer_name));
        if (observer_index < filtered_pose_publishers_.size()) {
          filtered_pose_publishers_[observer_index].publish(filtered_msg);
          // ROS_INFO("[PoseFiltration]: Published filtered poses for observer %s", observer_name.c_str());
        }
      }

      mrs_msgs::PoseWithCovarianceIdentified filterPose(const mrs_msgs::PoseWithCovarianceIdentified& raw_pose, int observer_id) {
        int target_id = raw_pose.id;

        // Add the raw pose to the history
        pose_history_map_[observer_id][target_id].push_back(raw_pose);

        // Remove oldest pose if history exceeds the size limit
        if (pose_history_map_[observer_id][target_id].size() > history_size_) {
          pose_history_map_[observer_id][target_id].pop_front();
        }

        // Calculate the average pose
        mrs_msgs::PoseWithCovarianceIdentified filtered_pose;
        filtered_pose.id = target_id;
        filtered_pose.pose.position.x = 0;
        filtered_pose.pose.position.y = 0;
        filtered_pose.pose.position.z = 0;
        filtered_pose.pose.orientation.x = 0;
        filtered_pose.pose.orientation.y = 0;
        filtered_pose.pose.orientation.z = 0;
        filtered_pose.pose.orientation.w = 0;
        filtered_pose.covariance = raw_pose.covariance; // For simplicity, keep covariance unchanged

        size_t num_samples = pose_history_map_[observer_id][target_id].size();
        for (const auto& past_pose : pose_history_map_[observer_id][target_id]) {
          filtered_pose.pose.position.x += past_pose.pose.position.x / num_samples;
          filtered_pose.pose.position.y += past_pose.pose.position.y / num_samples;
          filtered_pose.pose.position.z += past_pose.pose.position.z / num_samples;

          filtered_pose.pose.orientation.x += past_pose.pose.orientation.x / num_samples;
          filtered_pose.pose.orientation.y += past_pose.pose.orientation.y / num_samples;
          filtered_pose.pose.orientation.z += past_pose.pose.orientation.z / num_samples;
          filtered_pose.pose.orientation.w += past_pose.pose.orientation.w / num_samples;
        }

        // Normalize the quaternion to ensure valid orientation
        double norm = std::sqrt(
            std::pow(filtered_pose.pose.orientation.x, 2) +
            std::pow(filtered_pose.pose.orientation.y, 2) +
            std::pow(filtered_pose.pose.orientation.z, 2) +
            std::pow(filtered_pose.pose.orientation.w, 2)
        );
        filtered_pose.pose.orientation.x /= norm;
        filtered_pose.pose.orientation.y /= norm;
        filtered_pose.pose.orientation.z /= norm;
        filtered_pose.pose.orientation.w /= norm;

        return filtered_pose;
      }

      int extractIdFromName(const std::string& uav_name) {
        // Assuming UAV names are formatted as "uavX", where X is the ID
        return std::stoi(uav_name.substr(3));
      }
    };

} // namespace fish_model

int main(int argc, char** argv) {
  ros::init(argc, argv, "pose_filtration");

  ros::NodeHandle nh("~");
  filtration::PoseFiltration pf(nh);
  ROS_INFO("[PoseFiltration]: Pose Filtration node initiated");
  ros::spin();
  return 0;
}
