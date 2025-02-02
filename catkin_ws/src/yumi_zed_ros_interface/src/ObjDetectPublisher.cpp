
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// STD
#include <string>

// Custom
#include "yumi_zed_ros_interface/ObjDetectPublisher.hpp"

namespace object_detect {

    ObjDetectConverter::ObjDetectConverter(ros::NodeHandle& nodeHandle)
        : nodeHandle_(nodeHandle),
        instance_id_(0),
        label_("Person"),
        tfListener_(tfBuffer_),
        distance_thresh_(0.3),
        min_dist_(100.)
    {   
        string node_name = ros::this_node::getName();
        ROS_INFO_STREAM("Starting " << node_name << "node." );
        
        // Read ros parameters
        if (!readParameters()) {
            ROS_ERROR("Could not read parameters.");
            ros::requestShutdown();
            return;
        }


        // Check the topic of detection is alive
        // auto message = ros::topic::waitForMessage<zed_interfaces::ObjectsStamped>(ObjDetSubTopic_,
        //                                 nodeHandle_, ros::Duration(10.0));
        while(ros::ok() && ros::topic::waitForMessage<zed_interfaces::ObjectsStamped>(ObjDetSubTopic_,
                                        nodeHandle_, ros::Duration(3.0))) {
            // if (message) {
            //     break;  // Exit loop if message is received
            // } else {
                ROS_WARN_STREAM("No message received on " << ObjDetSubTopic_ <<  " . Waiting...");
                // auto message = ros::topic::waitForMessage<zed_interfaces::ObjectsStamped>(ObjDetSubTopic_,
                //                         nodeHandle_, ros::Duration(3.0));
            // }
        }
        
        ROS_INFO("Received Object message");
        // Init last_point_in_odom_ in 0,0,0
        last_point_in_odom_.x = 0.0;
        last_point_in_odom_.y = 0.0;
        last_point_in_odom_.z = 0.0;

        instances_list_ = vector<int>();

        // Subscribers
        obj_detect_subscriber_ = nodeHandle_.subscribe(ObjDetSubTopic_, 1,
                                        &ObjDetectConverter::objDetectCallback, this);
        
        // Publishers
        obj_detect_publisher_ = nodeHandle_.advertise<zed_interfaces::ObjectsStamped>(ObjDetPubTopic_, 10);

        // Services
        set_target_service_server_ = nodeHandle_.advertiseService(node_name +"/set_target_object",
                                                    &ObjDetectConverter::setTargetServiceCallback, this);
        
        set_target_service_client_ = nodeHandle_.serviceClient<zion_msgs::set_target_obj>(node_name +"/set_target_object");

        ROS_INFO_STREAM("Successfully launched " << node_name);

    }

    ObjDetectConverter::~ObjDetectConverter()
    {
    }

    bool ObjDetectConverter::readParameters()
    {
        if (!nodeHandle_.getParam("topics/obj_detect_sub_topic", ObjDetSubTopic_)) return false;
        if (!nodeHandle_.getParam("topics/obj_detect_pub_topic", ObjDetPubTopic_)) return false;
        if (!nodeHandle_.getParam("frames/input_frame", input_frame_)) return false;

        nodeHandle_.getParam("object/label", label_);
        nodeHandle_.getParam("object/instance_id", instance_id_);
        nodeHandle_.getParam("object/distance_thresh", distance_thresh_);

        ROS_INFO_STREAM("******* Parameters *******");
        ROS_INFO_STREAM("* Topics:");
        ROS_INFO_STREAM("  * obj_detect_sub_topic: " << ObjDetSubTopic_);
        ROS_INFO_STREAM("  * obj_detect_pub_topic: " << ObjDetPubTopic_);
        ROS_INFO_STREAM("* Frames:");
        ROS_INFO_STREAM("  * input_frame: " << input_frame_);

        ROS_INFO_STREAM("* Object:");
        ROS_INFO_STREAM("  * label " << label_);
        ROS_INFO_STREAM("  * instance_id " << instance_id_);
        ROS_INFO_STREAM("  * distance_thresh " << distance_thresh_);

        ROS_INFO_STREAM("**************************");
        return true;
    }

    void ObjDetectConverter::objDetectCallback(const zed_interfaces::ObjectsStamped& msg)
    {
        // Init msg & transformStamped
        zed_interfaces::ObjectsStamped obj_stamped_msg_;
        obj_stamped_msg_.header = msg.header;
        auto obj_candidate_ptr = msg.objects.end(); // Initialize to end()
        geometry_msgs::TransformStamped transformStamped;
        bool call_service = false;
        float min_dist = 100.;


        // TODO: chack if it is necessery to lookup for a specific transform -> I think its better
        // Transform camera point to odom frame
        try{
            transformStamped = tfBuffer_.lookupTransform("odom", input_frame_, ros::Time(0), ros::Duration(0.3));
        } catch (tf2::TransformException &ex) {
            ROS_WARN("%s",ex.what());
            // ros::Duration(1.0).sleep();
        }

        // Pull the specific instance from the objects list 
        for(auto obj = msg.objects.begin(); obj != msg.objects.end(); obj++){
            // Check valid label
            if(obj->label == label_){
                geometry_msgs::PointStamped camera_point;
                geometry_msgs::PointStamped odom_point;

                // Init point in camera frame
                camera_point.header.frame_id = input_frame_; 
                camera_point.header.stamp = msg.header.stamp;
                camera_point.point.x = obj->position[0];
                camera_point.point.y = obj->position[1];
                camera_point.point.z = obj->position[2];

                tf2::doTransform(camera_point, odom_point, transformStamped);

                // If id is the requiered one -> this is the right obj -> publish it!
                if(obj->instance_id==instance_id_){
                    last_point_in_odom_ = odom_point.point;
                    obj_candidate_ptr = obj;
                    obj_stamped_msg_.objects.push_back(*obj_candidate_ptr);
                    call_service = false;
                    break;
                // Else -> check if the last point in odom frame is the approxiamtley the current point in odom
                }else{
                    geometry_msgs::Point candidate_point_in_odom = odom_point.point;

                    // Compute euclidean distance
                    float distance = sqrt(pow(last_point_in_odom_.x - candidate_point_in_odom.x, 2) +
                                        pow(last_point_in_odom_.y - candidate_point_in_odom.y, 2) +
                                        pow(last_point_in_odom_.z - candidate_point_in_odom.z, 2));
                    
                    if(distance < min_dist){
                        min_dist = distance;
                        if (min_dist < distance_thresh_){
                            last_point_in_odom_ = candidate_point_in_odom;
                            obj_candidate_ptr = obj;
                            call_service = true;
                        }
                    }
                }
            }
        } // for

        // Use the service!
        if(call_service && obj_candidate_ptr != msg.objects.end()){
            zion_msgs::set_target_obj srv;
            srv.request.instance_id = obj_candidate_ptr->instance_id;
            srv.request.label = obj_candidate_ptr->label;

            // Change the requiered id, set the last point in odom to the current and publish the object
            if (set_target_service_client_.call(srv)) {
                if (srv.response.result) {
                    ROS_INFO_STREAM("Successfully updated target: " << srv.response.info);
                    obj_stamped_msg_.objects.push_back(*obj_candidate_ptr);
                } else {
                    ROS_WARN_STREAM("Failed to update target: " << srv.response.info);
                }
            } else {
                ROS_ERROR("Failed to call service set_target_object");
            }
        }

        // Publish
        obj_detect_publisher_.publish(obj_stamped_msg_);

    } // function callback

    bool ObjDetectConverter::setTargetServiceCallback(zion_msgs::set_target_objRequest& request,
                            zion_msgs::set_target_objResponse& response)
    {   

        if ( request.instance_id >=0 
            // && typeid(request.instance_id) == typeid(instance_id_)
            // && //typeid(request.label) == typeid(label_))
        ){
            label_ = request.label;
            instance_id_ = request.instance_id;
            response.info = "set target object to label: " + request.label + " | instance_id: " + to_string(request.instance_id);
            response.result = true;
        } else {
            response.info = "stay with object label: " + request.label + " | instance_id: " + to_string(request.instance_id);
            response.result = false;
        }

        // ROS_INFO_STREAM(response.info);
        return true;
    }


} /* namespace */