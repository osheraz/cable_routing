MODULE JACOBI_MODULE_T_ROB_L
VAR string next_state := "egm";

        VAR egmident egm_id;
        VAR egmstate egm_state;
        CONST egm_minmax egm_condition := [-0.03, 0.03];
        

PROC egm1()

        EGMActJoint egm_id
            \J1:=egm_condition \J2:=egm_condition \J3:=egm_condition
            \J4:=egm_condition \J5:=egm_condition \J6:=egm_condition \J7:=egm_condition
            \MaxSpeedDeviation:=1000.000000;
        EGMRunJoint egm_id, EGM_STOP_HOLD, \NoWaitCond \J1 \J2 \J3 \J4 \J5 \J6 \J7 \CondTime:=9e9;
        WaitDI JacobiEgmStop, 1;
        EGMStop egm_id, EGM_STOP_HOLD;

    ERROR
        IF ERRNO = ERR_UDPUC_COMM THEN
            TPWrite "[jacobi.driver] EGM communication timeout.";
            TRYNEXT;
        ENDIF
    IF next_state = "egm" THEN
        next_state := "egm";
    ENDIF
ENDPROC

PROC onReset()
	TPWrite "reset";
	EGMStop egm_id, EGM_STOP_HOLD;
ENDPROC

PROC onRestart()
	TPWrite "restart";
	EGMStop egm_id, EGM_STOP_HOLD;
ENDPROC

PROC onStart()
	TPWrite "start";
ENDPROC

PROC onStop()
	TPWrite "stop";
	EGMStop egm_id, EGM_STOP_HOLD;
ENDPROC


PROC main()
        
        initializeSGModule;
        EGMReset egm_id;
        EGMGetId egm_id;

        egm_state := EGMGetState(egm_id);
        TPWrite "EGM state: "\Num:=egm_state;

        IF egm_state <= EGM_STATE_CONNECTED THEN
            EGMSetupUC ROB_L, egm_id, "default", "ROB_L", \Joint;
        ENDIF
        
	WHILE next_state <> "exit" DO
		! TPWrite "[jacobi.driver] New state=" + next_state;
		egm1;
        
        command_input := cmd_GripperState_L;
        target_position_input := cmd_GripperPos_L;
        
        IF next_state <> "egm" THEN
            CallByVar next_state, 1;
        ENDIF
	ENDWHILE
ENDPROC

ENDMODULE
