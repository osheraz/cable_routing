MODULE TRobUtility

    !---------------------------------------------------------
    ! Program data
    !---------------------------------------------------------
    ! For information messages.
    LOCAL CONST string INDENTION        := "  ";
    LOCAL CONST string CONTEXT_UTILITY  := "[Utility]: ";
    LOCAL CONST string CONTEXT_SG       := "[SmartGripper]: ";

    !---------------------------------------------------------
    ! Auxiliary procedures
    !---------------------------------------------------------

    PROC printUtilityMessage(string message)
        printInfoMessage 0, CONTEXT_UTILITY, message;
    ENDPROC

    PROC printSGMessage(string message)
        printInfoMessage 0, CONTEXT_SG, message;
    ENDPROC

    LOCAL PROC printInfoMessage(num indention_level, string context, string message)
        VAR string temp_indention := "";

        IF(indention_level > 0) THEN
            FOR i FROM 0 TO indention_level - 1 DO
                temp_indention := temp_indention + INDENTION;
            ENDFOR
        ENDIF

        TPWrite temp_indention + context + message;
    ENDPROC

    PROC saturateValue(VAR num value, num minimum, num maximum)
        IF value < minimum THEN
            value := minimum;
        ELSEIF value > maximum THEN
            value := maximum;
        ENDIF
    ENDPROC

ENDMODULE