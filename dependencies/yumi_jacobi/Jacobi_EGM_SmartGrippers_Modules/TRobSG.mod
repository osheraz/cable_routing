MODULE TRobSG
!=======================================================================================================================
! Copyright (c) 2016, ABB Schweiz AG
! All rights reserved.
!
! Redistribution and use in source and binary forms, with
! or without modification, are permitted provided that
! the following conditions are met:
!
!    * Redistributions of source code must retain the
!      above copyright notice, this list of conditions
!      and the following disclaimer.
!    * Redistributions in binary form must reproduce the
!      above copyright notice, this list of conditions
!      and the following disclaimer in the documentation
!      and/or other materials provided with the
!      distribution.
!    * Neither the name of ABB nor the names of its
!      contributors may be used to endorse or promote
!      products derived from this software without
!      specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
! ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
! LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
! THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!=======================================================================================================================

    !-------------------------------------------------------------------------------------------------------------------
    !
    ! Module: TRobSG [Autoloaded by the StateMachine AddIn]
    !
    ! Description:
    !   Provides support of using the following SmartGripper features:
    !   - Initialization of the gripper.
    !   - Calibration of the gripper.
    !   - Move the gripper.
    !   - Grip inwards or outwards.
    !   - Turn on/off vacuum.
    !   - Turn on/off blow.
    !
    ! Author: Jon Tjerngren (jon.tjerngren@se.abb.com)
    !
    ! Version: 1.1
    !
    !-------------------------------------------------------------------------------------------------------------------

    !---------------------------------------------------------
    ! Records
    !---------------------------------------------------------
    LOCAL RECORD SGSettings
        num max_speed;       ! Allowed maximum speed [mm/s] for the gripper.
        num hold_force;      ! Expected force [N] used for gripping.
        num physical_limit;  ! The physical limit [mm] (if the gripper should operate in a smaller travel range).
    ENDRECORD

    !---------------------------------------------------------
    ! Data that an external system can set,
    ! for example via Robot Web Services (RWS)
    !---------------------------------------------------------
    ! Settings to the arguments used in the SmartGripper instructions.
    LOCAL VAR SGSettings settings;

    ! Input specifying the desired gripper command (supported commands specified below).
    VAR num command_input;

    ! The targeted position [mm] that the gripper should move to (used in the command "COMMAND_MOVE_TO").
    VAR num target_position_input;

    !---------------------------------------------------------
    ! Program data
    !---------------------------------------------------------
    ! The SmartGripper commands supported by this module.
    LOCAL CONST num COMMAND_NONE         := 0;
    LOCAL CONST num COMMAND_INITIALIZE   := 1;
    LOCAL CONST num COMMAND_CALIBRATE    := 2;
    LOCAL CONST num COMMAND_MOVE_TO      := 3;
    LOCAL CONST num COMMAND_GRIP_IN      := 4;
    LOCAL CONST num COMMAND_GRIP_OUT     := 5;
    LOCAL CONST num COMMAND_BLOW_ON_1    := 6;
    LOCAL CONST num COMMAND_BLOW_ON_2    := 7;
    LOCAL CONST num COMMAND_BLOW_OFF_1   := 8;
    LOCAL CONST num COMMAND_BLOW_OFF_2   := 9;
    LOCAL CONST num COMMAND_VACUUM_ON_1  := 10;
    LOCAL CONST num COMMAND_VACUUM_ON_2  := 11;
    LOCAL CONST num COMMAND_VACUUM_OFF_1 := 12;
    LOCAL CONST num COMMAND_VACUUM_OFF_2 := 13;

    ! Limits that the SmartGripper should adhere to:
    ! - Speed [mm/s].
    ! - Force [N].
    ! - Physical travel range [mm].
    LOCAL CONST num MAX_SPEED          := 25;
    LOCAL CONST num MAX_FORCE          := 20;
    LOCAL CONST num MAX_PHYSICAL_LIMIT := 25;
    
    ! Interrupt numbers.
    LOCAL VAR intnum intnum_run_sg_routine;
      
    !----------------------------
    ! Tool data components
    ! from the product manual
    !----------------------------
    CONST bool SG_ROBHOLD := TRUE;
    
    ! Default tool frames.
    CONST pose SG_FRAME_SERVO    := [[0,0,84],[1,0,0,0]];
    CONST pose SG_FRAME_FINGERS  := [[0,0,114.2],[1,0,0,0]];
    CONST pose SG_FRAME_VACUUM_1 := [[63.5,18.5,37.5],[0.707106781,0,0.707106781,0]];
    CONST pose SG_FRAME_VACUUM_2 := [[-63.5,18.5,37.5],[0,-0.707106781,0,0.707106781]];
    CONST pose SG_FRAME_CAMERA   := [[-7.3,28.3,35.1],[0.5,-0.5,0.5,0.5]];
    
    ! Load data (without default fingers, suction cup(s), and filter(s)).
    CONST loaddata SG_LOAD_SERVO_MIN               := [0.215,[8.7,12.3,49.2],[1,0,0,0],0.00017,0.00020,0.00008];
    CONST loaddata SG_LOAD_SERVO_VACUUM_MIN        := [0.226,[8.9,12.3,48.7],[1,0,0,0],0.00017,0.00020,0.00008];
    CONST loaddata SG_LOAD_SERVO_VACUUM_VACUUM_MIN := [0.250,[7.4,12.4,44.8],[1,0,0,0],0.00020,0.00024,0.00011];
    CONST loaddata SG_LOAD_SERVO_CAMERA_MIN        := [0.229,[7.9,12.4,48.7],[1,0,0,0],0.00017,0.00019,0.00008];
    CONST loaddata SG_LOAD_SERVO_CAMERA_VACUUM_MIN := [0.240,[8.2,12.5,48.1],[1,0,0,0],0.00018,0.00020,0.00009];
    
    ! Load data (with default fingers, suction cup(s), and filter(s)).
    ! - Default fingers weighs 15 gram per set.
    ! - Standard suction cups and filters weighs 7.5 gram per set.
    CONST loaddata SG_LOAD_SERVO               := [0.230,[8.2,11.7,52.0],[1,0,0,0],0.00021,0.00024,0.00009];
    CONST loaddata SG_LOAD_SERVO_VACUUM        := [0.248,[8.6,11.7,52.7],[1,0,0,0],0.00021,0.00024,0.00009];
    CONST loaddata SG_LOAD_SERVO_VACUUM_VACUUM := [0.280,[7.1,11.9,47.3],[1,0,0,0],0.00025,0.00029,0.00012];
    CONST loaddata SG_LOAD_SERVO_CAMERA        := [0.244,[7.5,11.8,52.7],[1,0,0,0],0.00021,0.00023,0.00008];
    CONST loaddata SG_LOAD_SERVO_CAMERA_VACUUM := [0.262,[7.8,11.9,50.7],[1,0,0,0],0.00022,0.00024,0.00009];

    !---------------------------------------------------------
    ! Primary procedures
    !---------------------------------------------------------
    PROC initializeSGModule()
        ! Set default settings for the SmartGripper instructions' arguments.
        settings.max_speed      := MAX_SPEED;
        settings.hold_force     := MAX_FORCE*0.5;
        settings.physical_limit := MAX_PHYSICAL_LIMIT;

        ! Set default inputs.
        command_input         := COMMAND_NONE;
        target_position_input := 0;

        ! Setup an interrupt signal
!        IDelete intnum_run_sg_routine;
!        CONNECT intnum_run_sg_routine WITH handleRunSGRoutine1;
!        ISignalDI RUN_SG_ROUTINE, HIGH, intnum_run_sg_routine;
    ENDPROC

    PROC handleRunSGRoutine1()
        IF command_input <> COMMAND_NONE THEN
            runSGRoutine;
        ENDIF
    ENDPROC

    !---------------------------------------------------------
    ! Auxiliary procedures
    !---------------------------------------------------------
    LOCAL PROC runSGRoutine()
        saturateArguments;

        printSGMessage mapCommandToString();

        IF RobOS() THEN
            TEST command_input
                CASE COMMAND_NONE:       ! Do nothing.
                CASE COMMAND_INITIALIZE: g_Init \maxSpd    := settings.max_speed
                                                \holdForce := settings.hold_force
                                                \phyLimit  := settings.physical_limit;
                CASE COMMAND_CALIBRATE:  g_Calibrate \Jog;
                DEFAULT:
                    IF g_IsCalibrated() THEN
                        TEST command_input
                            CASE COMMAND_MOVE_TO:      g_MoveTo target_position_input \NoWait;
                            CASE COMMAND_GRIP_IN:      g_GripIn \holdForce:=settings.hold_force \NoWait;
                            CASE COMMAND_GRIP_OUT:     g_GripOut \holdForce:=settings.hold_force \NoWait;
                            CASE COMMAND_BLOW_ON_1:    g_BlowOn1;
                            CASE COMMAND_BLOW_ON_2:    g_BlowOn2;
                            CASE COMMAND_BLOW_OFF_1:   g_BlowOff1;
                            CASE COMMAND_BLOW_OFF_2:   g_BlowOff2;
                            CASE COMMAND_VACUUM_ON_1:  g_VacuumOn1;
                            CASE COMMAND_VACUUM_ON_2:  g_VacuumOn2;
                            CASE COMMAND_VACUUM_OFF_1: g_VacuumOff1;
                            CASE COMMAND_VACUUM_OFF_2: g_VacuumOff2;
                            DEFAULT:                   ! Do nothing.
                        ENDTEST
                    ELSE
                        printSGMessage "Gripper is not calibrated";
                    ENDIF
            ENDTEST
        ELSE
            printSGMessage "Virtual controller (gripper is not simulated)";
        ENDIF

        command_input := COMMAND_NONE;
    ENDPROC

    LOCAL PROC saturateArguments()
        ! Saturate the maximum speed, hold force and physical limit.
        saturateValue settings.max_speed, 0, MAX_SPEED;
        saturateValue settings.hold_force, 0, MAX_FORCE;
        saturateValue settings.physical_limit, 0, MAX_PHYSICAL_LIMIT;

        ! Saturate the targeted position.
        saturateValue target_position_input, 0, settings.physical_limit;
        ! Saturate the command enum
        saturateValue command_input, 0, 13;
    ENDPROC

    !---------------------------------------------------------
    ! Auxiliary functions
    !---------------------------------------------------------
    LOCAL FUNC string mapCommandToString()
        VAR string result;
        result := "Received command " + NumToStr(command_input, 0) + " (";

        TEST command_input
            CASE COMMAND_NONE:         result := result + "No command";
            CASE COMMAND_INITIALIZE:   result := result + "Initialize gripper";
            CASE COMMAND_CALIBRATE:    result := result + "Calibrate gripper";
            CASE COMMAND_MOVE_TO:      result := result + "Move gripper to " + NumToStr(target_position_input, 0);
            CASE COMMAND_GRIP_IN:      result := result + "Grip inwards";
            CASE COMMAND_GRIP_OUT:     result := result + "Grip outwards";
            CASE COMMAND_BLOW_ON_1:    result := result + "Turn on blow 1";
            CASE COMMAND_BLOW_ON_2:    result := result + "Turn on blow 2";
            CASE COMMAND_BLOW_OFF_1:   result := result + "Turn off blow 1";
            CASE COMMAND_BLOW_OFF_2:   result := result + "Turn off blow 2";
            CASE COMMAND_VACUUM_ON_1:  result := result + "Turn on vacuum 1";
            CASE COMMAND_VACUUM_ON_2:  result := result + "Turn on vacuum 2";
            CASE COMMAND_VACUUM_OFF_1: result := result + "Turn off vacuum 1";
            CASE COMMAND_VACUUM_OFF_2: result := result + "Turn off vacuum 2";
            DEFAULT:                   result := result + "Unknown command";
        ENDTEST

        result := result + ")";

        RETURN result;
    ENDFUNC
ENDMODULE
