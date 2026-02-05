MODULE RPW26_depalletize   
    CONST robtarget pHomePallet:=[[400.000142505,1600.000207592,1129.245823313],[0.000000065,0.000000056,1,0.000000039],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];    
    !CONST robtarget pAppPalletCorner:=[[0.000012553,-0.000028128,200.888399296],[-0.000000001,0.000000028,1,0.000000013],[-1,2,-3,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    !CONST robtarget pPalletCorner:=[[0,0,0],[0,0,1,0],[-1,1,-3,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pAppTable:=[[350.000028167,499.999645127,532.397104575],[0.00000004,0.707106809,0.707106754,0.000000015],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pTable:=[[350.000004826,499.999625609,420.999972108],[-0.000000007,0.707106819,0.707106743,0.000000021],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pPalletCornerCamera:=[[-138.151003258,803.514794279,3061.000069285],[0.000000654,-0.00000004,-0.000000051,1],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    VAR robtarget target; !target that the camera creates for a pick-up point.
    VAR num gToolSelect;  ! 1 = big, 2 = small

    
    
    ! Tools and bases
    PERS tooldata TCP_small:=[TRUE,[[-240.64,-73.146,245],[1,0,0,0]],[10,[0,0,245],[1,0,0,0],0,0,0]];    
    PERS tooldata TCP_big:=[TRUE,[[-45,0,245],[1,0,0,0]],[10,[0,0,245],[1,0,0,0],0,0,0]];
    
    PERS wobjdata wobj_camera:=[FALSE,TRUE,"",[[1343,170,2615],[0,1,0,0]],[[0,0,0],[1,0,0,0]]]; ! NOTE: Camera Z+ points DOWN. Negative Offs Z moves UP. (mimicking camera pos)
    PERS wobjdata wobj_pallet:=[FALSE,TRUE,"",[[1204.849,-633.515,-446],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
    PERS wobjdata wobj_table:=[FALSE,TRUE,"",[[667.645,965.791,80],[0.707106781,0,0,0.707106781]],[[0,0,0],[1,0,0,0]]];
  
PROC main()
    VAR num fkAnswer;
    VAR errnum errVar;
    
    MoveJ PhomePallet,v500,z100,TCP_big\WObj:=wobj_pallet;
    TPWrite "System ready.";
    

    WHILE TRUE DO
        TPWrite "Press button on pendant when new box is placed";

        ! Wait for operator confirmation using FlexPendant function keys
        TPReadFK fkAnswer, "New box placed. Pick next?", stEmpty, stEmpty, stEmpty, "Yes", "No"
            \MaxTime:= 600   
            \BreakFlag:=errVar;

        ! If user pressed Yes (function key 4) or DI break, proceed
        IF fkAnswer = 4 OR errVar = ERR_TP_DIBREAK THEN
            TPWrite "Operator confirmed. Requesting box...";
            pickandplace;
            TPWrite "Cycle complete.";
        ENDIF

        ! If user pressed No (function key 5), just loop back and wait again
        IF fkAnswer = 5 THEN
            TPWrite "Waiting for next box...";
        ENDIF

    ENDWHILE
ENDPROC     
    
PROC getTargetfromVision()
    ! ------------------------
    ! Variables
    ! ------------------------
    VAR socketdev sock;
    VAR string response;
    VAR string r_x;
    VAR string r_y;
    VAR string r_z;
    VAR string r_th;
    VAR string r_n;
    VAR num d_x;
    VAR num d_y; 
    VAR num d_z;
    VAR num d_th;
    VAR num d_n;
    VAR bool ok;
    VAR num point1;
    VAR num point2;
    VAR num point3;   
    VAR num point4;
    
    ! ------------------------
    ! Create and connect socket
    ! ------------------------
    SocketClose sock;
    SocketCreate sock;
    TPWrite "Connecting to vision server...";
    SocketConnect sock, "192.168.0.4", 4500;  ! <-- PC IP and port
    TPWrite "Connected!";

    ! ------------------------
    ! Send request for box coordinates
    ! ------------------------
    SocketSend sock \Str := "CAPTURE";
    TPWrite "Request sent to vision server";

    ! ------------------------
    ! Receive response
    ! ------------------------
    SocketReceive sock \Str := response;
    TPWrite "Raw response: " + response;

    ! ------------------------
    ! Close socket
    ! ------------------------
    SocketClose sock;
   ! ------------------------
    ! Parse semicolon-separated response: x;y;z;theta_z
    ! ------------------------
    point1 := StrFind(response, 1, ";");
    point2 := StrFind(response, point1 + 1, ";");
    point3 := StrFind(response, point2 + 1, ";");
    point4 := StrFind(response, point3 + 1, ";");
    
    r_x  := StrPart(response, 1, point1 - 1);
    r_y  := StrPart(response, point1 + 1, point2 - point1 - 1);
    r_z  := StrPart(response, point2 + 1, point3 - point2 - 1);
    r_th := StrPart(response, point3 + 1, point4 - point3 - 1);
    r_n  := StrPart(response, point4 + 1, StrLen(response) - point4);


    TPWrite "Parsed strings: X=" + r_x + " Y=" + r_y + " Z=" + r_z + " ThetaZ=" + r_th;

    ! ------------------------
    ! Convert strings to numbers
    ! ------------------------
    ok := StrToVal(r_x, d_x);
    ok := StrToVal(r_y, d_y);
    ok := StrToVal(r_z, d_z);
    ok := StrToVal(r_th, d_th);
    ok := StrToVal(r_n, d_n);
    gToolSelect := d_n;
    
    IF NOT ok THEN !Check that conversion ok
        TPWrite "ERROR: Vision number conversion failed";
        RETURN;
    ENDIF

    TPWrite "Coordinates as numbers: X=" + NumToStr(d_x,2) +
            " Y=" + NumToStr(d_y,2) +
            " Z=" + NumToStr(d_z,2) +
            " ThetaZ=" + NumToStr(d_th,2);

    ! ------------------------
    ! Build robtarget for planar pick (rotation only around Z)
    ! ------------------------
    
    target.trans := [d_x, d_y, d_z];
    target.rot   := OrientZYX(d_th,0,0);  ! only Z rotation
    target.robconf  := [-1,0,-1,0];
    target.extax := [9E9,9E9,9E9,9E9,9E9,9E9];

ENDPROC

PROC pickandplace()
        MoveJ PhomePallet,v500,z100,TCP_big\WObj:=wobj_pallet;
        
        gToolSelect := 0;
        getTargetfromVision;
        
        !Pick Moves
        IF gToolSelect = 1 THEN !Big TCP         
            MoveL Offs(target, 0, 0, -200), v500, z50, TCP_big \WObj:=wobj_camera;
            MoveL target,v200,z0,TCP_big\WObj:=wobj_camera;
            WaitTime 0.5;
            SetDO do1_Tool1_Close, 0; !Make sure tools can be turned on
            SetDO do3_Tool2_Close, 0;
            SetDO do2_Tool1_Open, 1; !for big boxes, both areas are turned on
            SetDO do4_Tool2_Open, 1;
            
            WaitTime 0.5;
            MoveL Offs(target, 0, 0, -200), v500, z50, TCP_big \WObj:=wobj_camera;     
            MoveJ PhomePallet,v500,z100,TCP_big\WObj:=wobj_pallet;           
        
        ELSEIF gToolSelect = 2 THEN !Small TCP
            MoveL Offs(target, 0, 0, -200), v500, z50, TCP_small \WObj:=wobj_camera;
            MoveL target,v200,z0,TCP_small\WObj:=wobj_camera;
            
            WaitTime 0.5;
            SetDO do3_Tool2_Close, 0; !Make sure tools can be turned on
            SetDO do4_Tool2_Open, 1; !For small boxes, only small portion turned on
            WaitTime 0.5;
            
            MoveL Offs(target, 0, 0, -200), v500, z50, TCP_small \WObj:=wobj_camera;     
            MoveJ PhomePallet,v500,z100,TCP_small\WObj:=wobj_pallet;             
        
        ELSE
            TPWrite "ERROR: Invalid tool number from vision";
            RETURN;
        ENDIF 
        
        !Place move
        IF gToolSelect = 1 THEN  !Big TCP
            MoveJ Offs(pTable,0,0,200), v500, z50, TCP_big\WObj:=wobj_table;
            MoveL pTable,v200,z0,TCP_big\WObj:=wobj_table;
            
            WaitTime 0.5;
            SetDO do2_Tool1_Open, 0; 
            SetDO do4_Tool2_Open, 0;
            SetDO do1_Tool1_Close, 1; 
            SetDO do3_Tool2_Close, 1;            
            
            WaitTime 0.5;
            MoveL Offs(pTable,0,0,200), v500, z50, TCP_big\WObj:=wobj_table;
            MoveJ PhomePallet,v500,z100,TCP_big\WObj:=wobj_pallet;
            
        ELSE !Small TCP
            MoveJ Offs(pTable,0,0,200), v500, z50, TCP_small\WObj:=wobj_table;
            MoveL pTable,v200,z0,TCP_small\WObj:=wobj_table;
            
            WaitTime 0.5;
            SetDO do4_Tool2_Open, 0;
            SetDO do3_Tool2_Close, 1;
            
            WaitTime 0.5;
            MoveL Offs(pTable,0,0,200), v500, z50, TCP_small\WObj:=wobj_table;
            MoveJ PhomePallet,v500,z100,TCP_small\WObj:=wobj_pallet;
        ENDIF
    ENDPROC
    


ENDMODULE