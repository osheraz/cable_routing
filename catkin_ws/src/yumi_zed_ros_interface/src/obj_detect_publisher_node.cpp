
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ros/ros.h>
#include "yumi_zed_ros_interface/ObjDetectPublisher.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "obj_detect_publisher_node");
  ros::NodeHandle nodeHandle("~");

  object_detect::ObjDetectConverter objdet(nodeHandle);

  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();

  return 0;
}