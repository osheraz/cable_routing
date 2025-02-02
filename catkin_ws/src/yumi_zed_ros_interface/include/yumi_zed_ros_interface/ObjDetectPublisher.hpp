// Copyright 2024 Nimrod Curtis

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// ROS
#include <ros/ros.h>
#include <zed_interfaces/ObjectsStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// Custom
#include <zion_msgs/set_target_objRequest.h>
#include <zion_msgs/set_target_objResponse.h>
#include <zion_msgs/set_target_obj.h>

using namespace std;

namespace object_detect {

/*!
 * Main class for the node to handle the ROS interfacing.
 */
class ObjDetectConverter
{
  public:
    /*!
   * Constructor.
   * @param nodeHandle the ROS node handle.
   */
  ObjDetectConverter(ros::NodeHandle& nodeHandle);

  /*!
   * Destructor.
   */
    virtual ~ObjDetectConverter();

  private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
    bool readParameters();

    /*!
   * ROS topic callback method.
   * @param message the received message.
   */
    void objDetectCallback(const zed_interfaces::ObjectsStamped& msg);

    /*!
     * ROS service server callback for setting the object label and instance to track for.
     * @param request request msg.
     * @param response the provided response.
     * @return true if successful, false otherwise.
     */
    bool setTargetServiceCallback(zion_msgs::set_target_objRequest& request,
                        zion_msgs::set_target_objResponse& response);

  //! ROS node handle.
    ros::NodeHandle& nodeHandle_;

  //! ROS topic subscriber.
    ros::Subscriber obj_detect_subscriber_;

  //! ROS topic publisher.
    ros::Publisher obj_detect_publisher_;

  //! ROS service server & client.
    ros::ServiceServer set_target_service_server_;  
    ros::ServiceClient set_target_service_client_;

  //! ROS tf2 helpers.
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;
  
  //! ROS topic name to subscribe to.
    std::string goalTopic_;
    std::string ObjDetSubTopic_;
    std::string ObjDetPubTopic_;

  // Node variables
    int instance_id_;  // object instance
    string label_;     // object label
    geometry_msgs::Point last_point_in_odom_;
    float distance_thresh_;
    string input_frame_; 
    vector<int> instances_list_;
    float min_dist_;
    int instance_min_dist;
};

} /* namespace */