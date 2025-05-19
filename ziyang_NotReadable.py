route_cable(routing, primary_arm, display, dual_arm, save_viz)
│
├─► 1. 获取所有夹点 clips = self.board.get_clips()
│
├─► 2. 路径追踪与朝向计算
│     └─ self.update_cable_path(...)   # generate path_in_pixels, path_in_world, cable_orientations
│         ├─ self.trace_cable(...)  # 路径像素追踪
                1) find_nearest_white_pixel: find nearest 10 white pixels to Clip “A”
                    a) center_pixels_on_cable: find nearest 10 white pixels to Clip “A”
                2) filter the points close to all clips, then randomly find a start tracing point
                3) self.tracer.trace:
                    a) self.analytic_tracer.trace (in single_tracer.py): find 4 points close to the center pixel
                    b) self.tracer.trace: handloom #得到离散路径点
                        i) self._get_evenly_spaced_points   #等间隔采样
                        ii) self._trace: handloom #得到离散的最终路径点  与  TraceEnd.status class TraceEnd(Enum):
                                                                                                # EDGE = 1
                                                                                                # ENDPOINT = 2
                                                                                                # FINISHED = 3
                                                                                                # RETRACE = 4
                                                                                                # CLIP = 5
                        iii) May retrace when meet clip (status == TraceEnd.CLIP)  but always return TraceEnd.FINISHED?
          ├─ self.board.set_cable_path(path_in_pixels)?
├─► 3. 初始抓取
│     ├─ if not dual_arm:
│     │     └─ self.grasp_cable_node(...)  # 单手抓取
│     └─ else:
│           └─ self.dual_grasp_cable_node(...)  # 双手协同抓取
│               1) find proper grasp point and follow point 
│               2) extract their orientations on the cable
                3) self.robot.dual_hand_grasp  # grasp the cable using two hands (raise them a bit and move down in a low speed)

├─► 4. 抬高手臂避免缠绕
│     └─ robot.go_delta([0,0,0.04], [0,0,0.04])
│
├─► 5. 依次经过每个夹点，执行绕行与滑动
│     └─ for i in 1..len(routing)-2:
│           └─ self.route_around_clip(...)
│               ├─ calculate_sequence(...)  # 计算节点的绕行顺序 (3个方向)
│               ├─ for s in sequence:
│               │     └─ self.slideto_cable_node(...)  # 滑动到下一个点
                         1) self.plan_slide_to_cable_node
                            i) confirm start pixel (end effector) and end pixel (cable node direction)
                            ii) self.planner.plan_path: generate waypoints in pixels
                         2) self.execute_dual_slide_to_cable_node
                            i) generate second arm waypoints (a vector reference to the primary arm), adjust z, and y for collision avoidance 
                            ii) second arm translation to start point, then rotation
                                second arm move & rotation with 3 points
                                primary arm translation to start point, then rotation
                                loop through the waypoints: primary arm move & rotation, then second arm move & rotation
│               │     └─ 判断是否需要换手 need_regrasp(...)
                         右手为主手, 下一个CLIP在当前CLIP右侧, 则需要换手
                         左手为主手, 下一个CLIP在当前CLIP左侧, 则需要换手
│               │     └─ self.swap_arms(...)（如需换手）
                         1) self.perform_nearest_analytic_grasp_dual
                             i) self.get_nearest_analytic_grasp_point # 找到线缆上最近的抓取点
                             ii)self.robot.dual_hand_grasp
│           └─ update_routing_progress_px(...)  # 更新进度
│
├─► 6. 收尾动作
│     ├─ robot.open_grippers(secondary_arm)
│     ├─ robot.move_to_home(arm=secondary_arm)
│     ├─ robot.go_delta(...)  # 主手臂抬高
│     ├─ get_world_coord_from_pixel_coord(...)  # 终点坐标
│     └─ robot.single_hand_move(...)  # 主手臂移动到终点
│
└─► 7. 结束