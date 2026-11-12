"""
This file contains all reward functions that were tested for this work.
Some may be outdated from a compatibility perspective.

Author: Cemal Yilmat - 2026
"""

def reward_function(state, _q0_prev, torque, torque_prev, phase, state_koz, koz_violation_cnt, time_elapsed, termination, discount_factor, koz_margin_min_prev):
    q0_current = state[0]
    ang_vel_sat_x = state[4]
    ang_vel_sat_y = state[5]
    ang_vel_sat_z = state[6]
    q0_prev = _q0_prev
    torque_1 = torque[0]
    torque_2 = torque[1]
    torque_3 = torque[2]
    torque_1_prev = torque_prev[0]
    torque_2_prev = torque_prev[1]
    torque_3_prev = torque_prev[2]
    koz_margin_min = np.min(state_koz[:, 0]) # For KOZ related reward calculation, use the minimum margin angle of all KOZs.
    closest_koz_idx = np.argmin(state_koz[:, 0])
    koz_direction_closest = state_koz[closest_koz_idx, 1:4] # The direction of the closest KOZ in inertial frame
    koz_margins_violated = [margin for margin in state_koz[:, 0] if margin <= 0]
    
    # Clamp q0 values to [-1, 1] to prevent acos() domain errors (NaN) with large torques
    # Using min/max instead of np.clip for numba compatibility with scalars
    q0_current = min(max(abs(q0_current), -1.0), 1.0)
    q0_prev = min(max(abs(q0_prev), -1.0), 1.0)
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    err_phi_current = err_phi_current * 180.0 / np.pi
    err_phi_prev = err_phi_prev * 180.0 / np.pi

    err_phi_delta = err_phi_prev - err_phi_current

    ang_vel_norm = calc_vector_norm(np.array([ang_vel_sat_x, ang_vel_sat_y, ang_vel_sat_z]))
    ang_vel = np.array([ang_vel_sat_x, ang_vel_sat_y, ang_vel_sat_z])

    boresight_unit_vector = np.array([1.0, 0.0, 0.0]) # The boresight vector in body frame is along the x-axis

    koz_margin_delta = koz_margin_min - koz_margin_min_prev

    r_total = 0
    USE_REWARD = "mod224ph2c2rev9" # mod224ph2c2g
    
    if USE_REWARD == "paper1":
        # Reward for reducing attitude error
        #r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

        # Penalty for high angular velocity (more than 0.1 rad/s)
        r2 = 0.0
        if ang_vel_sat_x > 0.1 or ang_vel_sat_y > 0.1 or ang_vel_sat_z > 0.1:
            r2 = -1

        # Reward for reducing attitude error and pointing accuracy
        r3 = 0.0
        if err_phi_current < 0.25:
            r3 = 1
        else:
            r3 = 0.5 * (1 - ((err_phi_current-0.25)/180.0)**0.6)

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        r_total = r2 + r3 + r4

    if USE_REWARD == "yang":

        r_err = np.exp(-err_phi_current/(0.14*360))
        r_torque = -0.05 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)/scale_torque_norm - 0.005 * (np.sqrt((torque_1-torque_1_prev)**2 + (torque_2-torque_2_prev)**2 + (torque_3-torque_3_prev)**2))
        r_acc = 0
        if err_phi_current < 0.25:
            r_acc = 9

        r_direction = 0
        if err_phi_current > err_phi_prev:
            r_direction = -1

        r_total = r_err + r_torque + r_acc + r_direction

    if USE_REWARD == "yangMod1":
        r_err = np.exp(-err_phi_current/(0.14*360.0))
        #r_torque = -0.05 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)/scale_torque_norm_yang - 0.005 * (np.sqrt((torque_1-torque_1_prev)**2 + (torque_2-torque_2_prev)**2 + (torque_3-torque_3_prev)**2))
        r_acc = 0
        if err_phi_current < 0.25:
            r_acc = 9

        r_direction = 0
        if err_phi_current > err_phi_prev:
            r_direction = -1

        r_total = r_err + r_acc + r_direction

    if USE_REWARD == "yangMod2":
        r_err = 0.5*np.exp(-err_phi_current/20.0)

        # Penalty for entering / being close to keep out zone
        r_koz = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r_koz = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r_koz = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r_koz = 0.0

        r_total = r_err + r_koz

    if USE_REWARD == "yangMod5":
        r_err = 0.0
        if koz_margin_min > 0:
            r_err = 0.5*np.exp(-err_phi_current/30.0)

        r_progress = 0.0
        if koz_margin_min > 0 and err_phi_delta > 0:
            r_progress = min(err_phi_delta, 0.2) * min(koz_margin_min, 0.2)

        r_koz_direction = 0.0
        if koz_margin_delta < 0:
            r_koz_direction = 200* koz_margin_delta * np.exp(-koz_margin_min * 30.0)
            r_koz_direction = max(r_koz_direction, -1.0)

    
        r_stall = -0.2 * np.exp(-(180-err_phi_current) / 200.0)

        # Penalty for entering / being close to keep out zone
        r_koz = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r_koz = -20 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_norm, 0.1)

            else:
                r_koz = -20 * np.exp(-koz_margin_min * 25.0) * max(ang_vel_norm, 0.1)

        # if time_elapsed < 50:
        #     print(f"[{time_elapsed:.1f}] err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(koz_margin_min*180/np.pi):.2f}, r_err {r_err:.3f}, r_progress {r_progress:.3f}, r_koz_direction {r_koz_direction:.3f}, r_koz {r_koz:.3f}")

        r_total = r_err + r_progress + r_koz_direction + r_koz + r_stall

    if USE_REWARD == "yangMod6":
        r_err = 0.0
        if koz_margin_min > 0:
            r_err = 0.5*np.exp(-err_phi_current/40.0)
            if err_phi_current < 1.0:
                r_err += 1

        r_progress = 0.0
        if koz_margin_min > 0:
            r_progress = 100 * err_phi_delta * np.exp(-err_phi_current)
            r_progress = min(r_progress, 0.5)

        r_koz_direction = 0.0
        if koz_margin_delta < 0:
            r_koz_direction = 200* koz_margin_delta * np.exp(-koz_margin_min * 15.0)


        # Penalty for entering / being close to keep out zone
        r_koz = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                ang_vel_eff_norm = np.sqrt(ang_vel_sat_y**2 + ang_vel_sat_z**2) # exclude x component as it is boresight vector.
                r_koz = -30 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_eff_norm, 0.05)

            else:
                r_koz = 0

        r_speed = 0.0
        if ang_vel_norm > 0.17:
            r_speed = -1.0

        
        print(f"[{time_elapsed:.1f}] err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(koz_margin_min*180/np.pi):.2f}, r_err {r_err:.3f}, r_progress {r_progress:.3f}, r_koz_direction {r_koz_direction:.3f}, r_koz {r_koz:.3f}")

        r_total = r_err + r_progress + r_koz_direction + r_koz + r_speed

    if USE_REWARD == "yangMod7":
        r_err = 0.0
        if koz_margin_min > 0:
            r_err = 0.5*np.exp(-err_phi_current/40.0)
            if err_phi_current < 1.0:
                r_err += 1

        r_progress = 0.0
        if koz_margin_min > 0:
            r_progress = 30 * err_phi_delta * np.exp(-err_phi_current / 2.0)
            r_progress = max(min(r_progress, 0.5), -0.5)

        r_koz_direction = 0.0
        if koz_margin_delta < 0:
            r_koz_direction = 200* koz_margin_delta * np.exp(-koz_margin_min * 15.0)


        # Penalty for entering / being close to keep out zone
        r_koz = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                ang_vel_eff_norm = np.sqrt(ang_vel_sat_y**2 + ang_vel_sat_z**2) # exclude x component as it is boresight vector.
                r_koz = -30 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_eff_norm, 0.05)

            else:
                r_koz = 0

        r_speed = 0.0
        if ang_vel_norm > 0.17:
            r_speed = -1.0

        
        print(f"[{time_elapsed:.1f}] err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(koz_margin_min*180/np.pi):.2f}, r_err {r_err:.3f}, r_progress {r_progress:.3f}, r_koz_direction {r_koz_direction:.3f}, r_koz {r_koz:.3f}")

        r_total = r_err + r_progress + r_koz_direction + r_koz + r_speed

    if USE_REWARD == "prak":
        # Reward for reducing attitude error
        r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

        # Bonus for high accuracy
        r3 = 0.0
        if err_phi_current < 0.25:
            r3 = 0.01  # bonus for reaching the goal
        else:
            r3 = -0.01

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            if koz_margin_min <= 0.0:
                r5 = -1.0
            else:
                r5 = -1.0*math.exp(-66.0*koz_margin_min)

        r_total = r1 + r3 + r4 + r5

    if USE_REWARD == "prakModAcc":
        # Reward for reducing attitude error
        r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            if phase == 2:
                r2 = 0.5
            else:
                r2 = 0.1
        # Gradial bonus for getting closer to target when already close to target
        elif err_phi_current < 5.0:
            r2 = 0.1*math.exp(-0.35*err_phi_current)
        # Penalty if farther away
        else:
            r2 = -0.01

        # Penalty for high angular velocity (more than 0.1 rad/s or 5.7 deg/s)
        r3 = 0.0
        if (ang_vel_sat_x > 0.1) or (ang_vel_sat_y > 0.1) or (ang_vel_sat_z > 0.1):
            r3 = -1.0

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0*math.exp(-10.0*koz_margin_min)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod2":
        """
        Goal: reduce violation rate.
        Result: very good.
        Note: pointing accuracy still does not recover.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.2:
            r2 = 0.02
        elif err_phi_current < 0.25:
            r2 = 0.02 * ((0.25-err_phi_current)/0.05)
        # Penalty if farther away
        else:
            r2 = -0.01


        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod22":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: Settling rate improved from 10% to 50%
        Note: bad reward design: if koz violation, no bonus for accuracy but violation penalty not hard enough.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == "phase 2":
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod221":
        """
        Goal: improve pointing accuracy when KOZ obs is not zero (compared to mod22)
        Result: 
        Note: Increased magnitude of pointing accuracy reward.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.2
            elif err_phi_current < 0.25:
                r2 = 0.2 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == "phase 2":
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod222":
        """
        Goal: recover pointing accuracy
        Result: 
        Note: basically a fix of mod221, where it had a reward jump at 0.25 deg attitude error.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.2 + 0.01
            elif err_phi_current < 0.25:
                r2 = 0.2 * ((0.25-err_phi_current)/0.05) + 0.01
            
        elif phase == "phase 2":
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.2 + 0.01
            elif err_phi_current < 0.25:
                r2 = 0.2 * ((0.25-err_phi_current)/0.05) + 0.01
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod223":
        """
        Goal: improve pointing accuracy when KOZ obs is not zero (compared to mod22)
        Result: 
        Note: Increased magnitude of pointing accuracy reward. Denser.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.2 + 0.01
            else:
                r2 = 0.2 * np.exp((-err_phi_current + 0.2) * 5.0) + 0.01
            
        elif phase == "phase 2":
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod223b":
        """
        Goal: improve pointing accuracy compared to mod223
        Result: 
        Note: increased weight
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.2
            else:
                r2 = 0.2 * np.exp((-err_phi_current + 0.2) * 20.0)
            
        elif phase == "phase 2":
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.2
            else:
                r2 = 0.2 * np.exp((-err_phi_current + 0.2) * 20.0)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224":
        """
        Goal: improve pointing accuracy when KOZ obs is not zero (compared to mod22)
        Result: 
        Note: Denser.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.2) * 1.0)
            
        elif phase == "phase 2":
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224b":
        """
        Goal: improve pointing accuracy compared to mod224
        Result: 
        Note: higher weight.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.2) * 5.0)
            
        elif phase == "phase 2":
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.2) * 5.0)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224c":
        """
        Goal: optimize
        Result: 
        Note: 
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = 0.1 * err_phi_delta
           
        # Phase 2
        else:      
            r1 = 0.1 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 1.0
            
        elif phase == "phase 2":
            if err_phi_current < 0.25:
                r2 = 1.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r5

    if USE_REWARD == "mod224c1":
        """
        Goal: optimize
        Result: 
        Note: 
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = 0.1 * err_phi_delta
            
        # Phase 2
        else:      
            r1 = 0.1 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 1.0
            
        elif phase == "phase 2":
            if err_phi_current < 0.25 and koz_violation_cnt == 0:
                r2 = 1.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r5

    if USE_REWARD == "mod224c2":
        """
        Goal: optimize
        Result: 
        Note: Penalty for each timestep where not settled
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = 0.1 * err_phi_delta
            
        # Phase 2
        else:      
            r1 = 0.1 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 1.0
            
        elif phase == "phase 2":
            if err_phi_current < 0.25 and koz_violation_cnt == 0:
                r2 = 1.0
            else:
                r2 = -0.01

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r5

    if USE_REWARD == "mod224c3":
        """
        Goal: optimize
        Result: 
        Note:
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = 0.1 * err_phi_delta
            
        # Phase 2
        else:      
            r1 = 0.1 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 1.0
            else:
                r2 = -0.01
            
        elif phase == "phase 2":
            if err_phi_current < 0.25:
                r2 = 1.0
            else:
                r2 = -0.01

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -2.0
            # Gradial penalty if outside
            else:
                r5 = -2.0 * np.exp(-koz_margin_min * 60.0)
        
        r_total = r1 + r2 + r5

    if USE_REWARD == "mod224c5":
        """
        Goal: optimize
        Result: 
        Note:
        """

        # Bonus for target proximity
        r1 = 0.1 * np.exp(-err_phi_current/(0.14*360))

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = 1.0

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d1":
        """
        Goal: optimize
        Result: 
        Note:
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = 1.0

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d2":
        """
        Goal: optimize
        Result: 
        Note: add proximity reward
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = 0.1
        else:
            r2 = 0.1 * np.exp(- (err_phi_current - 0.25)*5.0)

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d3":
        """
        Goal: optimize
        Result: 
        Note: different weight
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = 0.1
        else:
            r2 = 0.1 * np.exp(- (err_phi_current - 0.25)*1.0)

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d4":
        """
        Goal: optimize
        Result: 
        Note: acc bonus jump
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = 0.2
        else:
            r2 = 0.1 * np.exp(- (err_phi_current - 0.25)*1.0)

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d4a1":
        """
        Goal: optimize
        Result: 
        Note: higher possible safe progress
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.5)
                else:
                    r1 = 1.5 * err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = 0.2
        else:
            r2 = 0.1 * np.exp(- (err_phi_current - 0.25)*1.0)

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.5
            # Gradial penalty if outside
            else:
                r5 = -1.5 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d4a2":
        """
        Goal: optimize
        Result: 
        Note: no acc bonus jump, but dense.
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = -6553.6 * err_phi_current**8 + 0.2 # [0.1, 0.2]
        else:
            r2 = 0.1 * np.exp(- (err_phi_current - 0.25)*1.0)

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d4a3":
        """
        Goal: optimize
        Result: 
        Note: no acc bonus jump, but dense. different weight.
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = -6553.6 * err_phi_current**8 + 0.2 # [0.1, 0.2]
        else:
            r2 = 0.1 * np.exp(- (err_phi_current - 0.25)*0.2)

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d4b1":
        """
        Goal: optimize
        Result: 
        Note: only reward current error on desired pointing, otherwise only delta error.
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = -6553.6 * err_phi_current**8 + 0.2 # [0.1, 0.2]

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d4b2":
        """
        Goal: optimize
        Result: 
        Note: only reward current error on desired pointing (otherwise small penalty), otherwise only delta error.
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = -6553.6 * err_phi_current**8 + 0.2 # [0.1, 0.2]
        else:
            r2 = -0.05 * min((err_phi_current - 0.25) / 1.0, 1.0) # [-0.05, 0] penalty for farther away but capped.

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224d4c1":
        """
        Goal: optimize
        Result: 
        Note: increasing penalty with time while not desired pointing.
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = -err_phi_current + 0.35 # [0.1, 0.35]

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)

        # Time increasing penalty for being outside desired accuracy
        r4 = -time_elapsed * 0.0001 * err_phi_current
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r4 + r5

        
    if USE_REWARD == "mod224d5":
        """
        Goal: optimize
        Result: 
        Note: acc bonus higher jump
        """

        # Bonus for reducing attitude
        r1 = 0.0
        if phase == "phase 1":
            r1 = err_phi_delta
        else:
            if koz_margin_min > 0:
                if err_phi_delta >= 0:
                    r1 = min(err_phi_delta, 1.0) * min(koz_margin_min, 1.0)
                else:
                    r1 = err_phi_delta
            else:
                r1 = 0.0

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            r2 = 0.5
        else:
            r2 = 0.1 * np.exp(- (err_phi_current - 0.25)*1.0)

        # Torque penalty
        r3 = -0.01 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)
            
        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty if outside
            else:
                r5 = -1.0 * np.exp(-koz_margin_min * 50.0)
        
        r_total = r1 + r2 + r3 + r5

    if USE_REWARD == "mod224ph2a":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: changed pointing acc to 0.25. +/- delta err same weight, - delta less weight if close to KOZ. pointing bonus even when violated.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.02
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.2) * 1.0)
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.02
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.2) * 1.0)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224ph2b":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: changed pointing acc to 0.25. +/- delta err same weight, - delta less weight if close to KOZ. pointing bonus even when violated.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            

        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.02
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.2) * 1.0)

        elif phase == "phase 2":
            if err_phi_current < 0.25:
                r2 = 0.02
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.2) * 1.0)


        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0


        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224ph2c":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: higher reward for desired pointing.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current <= 0.25:
                r2 = 0.06
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.25) * 1.0)
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if err_phi_current <= 0.25:
                r2 = 0.06
            else:
                r2 = 0.02 * np.exp((-err_phi_current + 0.25) * 1.0)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224ph2c2":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: pointing reward no jump.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224ph2c2rev1":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Revision 1: high velocity penalty, KOZ penalty scales with velocity.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -20 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_norm, 0.05)
            # No penalty if farther away
            else:
                r5 = 0.0

        r_speed = 0.0
        if ang_vel_norm > 0.0874: # around 5 deg/s
            r_speed = -1.0

        
        r_total = r1 + r2 + r4 + r5 + r_speed

    if USE_REWARD == "mod224ph2c2rev2":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Revision 2: Make KOZ penalty continuous starting close. Increase KOZ penalty weight. Increase pointing weight. Increase torque weight.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -200 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_norm, 0.05)
            # No penalty if farther away
            else:
                r5 = -200 * np.exp(-koz_margin_min * 219.0) * max(ang_vel_norm, 0.05)

        r_speed = 0.0
        if ang_vel_norm > 0.0874: # around 5 deg/s
            r_speed = -1.0

        
        r_total = r1 + r2 + r4 + r5 + r_speed

    if USE_REWARD == "mod224ph2c2rev4":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Revision 3: Compared to rev3: add koz directional, lower pointing weight.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0

            if r1 < 0:
                r1 = r1 * 1.1
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-((err_phi_current / 1.5)))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-((err_phi_current / 1.5)))

            r2 += - err_phi_current / 1000.0
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            for margin in koz_margins_violated:
                # Maximum penalty inside of KOZ
                r5 += -20 * np.exp(-margin * 6.0) * max(ang_vel_norm, 0.05)

            if koz_margin_min > 0:
                r5 = -20 * np.exp(-koz_margin_min * 219.0) * max(ang_vel_norm, 0.05)

        r_koz_direction = 0.0
        if koz_margin_delta < 0 and koz_margin_min > 0:
            r_koz_direction = 200* koz_margin_delta * np.exp(-koz_margin_min * 20.0)

        r_speed = 0.0
        if ang_vel_norm > 0.0874: # around 5 deg/s
            r_speed = -1.0

        r_total = r1 + r2 + r4 + r5 + r_koz_direction + r_speed

    if USE_REWARD == "mod224ph2c2rev5":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Revision 5: more close to og again.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-(err_phi_current / 1.5))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-(err_phi_current / 1.5))
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            for margin in koz_margins_violated:
                # Maximum penalty inside of KOZ
                r5 += -20 * np.exp(-margin * 6.0) * max(ang_vel_norm, 0.05)
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            if 0 < koz_margin_min < 0.17:
                r5 = -5 * (1.0 - koz_margin_min/0.17) * max(ang_vel_norm, 0.05)

        r_speed = 0.0
        if ang_vel_norm > 0.0874: # around 5 deg/s
            r_speed = -1.0

        r_koz_direction = 0.0
        if koz_margin_delta < 0 and koz_margin_min > 0:
            r_koz_direction = 200* koz_margin_delta * np.exp(-koz_margin_min * 20.0)

        r_total = r1 + r2 + r4 + r5 + r_speed + r_koz_direction

    if USE_REWARD == "mod224ph2c2rev6":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Revision 6: compareed to rev5: different speed penalty.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-(err_phi_current / 1.5))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-(err_phi_current / 1.5))
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            for margin in koz_margins_violated:
                # Maximum penalty inside of KOZ
                r5 += -20 * np.exp(-margin * 6.0) * max(ang_vel_norm, 0.05)
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            if 0 < koz_margin_min < 0.17:
                r5 = -5 * (1.0 - koz_margin_min/0.17) * max(ang_vel_norm, 0.05)

        r_speed = 0.0
        if ang_vel_norm > 0.0874 and koz_margin_min < 0.26: # around 5 deg/s
            r_speed = 60 * (0.0874-ang_vel_norm) * (0.26-koz_margin_min)

        r_koz_direction = 0.0
        if koz_margin_delta < 0 and koz_margin_min > 0:
            r_koz_direction = 200* koz_margin_delta * np.exp(-koz_margin_min * 20.0)

        r_total = r1 + r2 + r4 + r5 + r_speed + r_koz_direction

    if USE_REWARD == "mod224ph2c2rev7":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Compared to og: only scale KOZ penalty with speed.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -20 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_norm, 0.05)
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -20.0 * (1.0 - koz_margin_min/0.17) * max(ang_vel_norm, 0.05)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224ph2c2rev8":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Compared to rev7: increased pointing reward.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-err_phi_current / 1.5)
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-err_phi_current / 1.5)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -20 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_norm, 0.05)
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -20.0 * (1.0 - koz_margin_min/0.17) * max(ang_vel_norm, 0.05)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224ph2c2rev9":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: Compared to rev8: less weight reducing attitude error
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta * 0.1
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0

            r1 = 0.1 * r1
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-err_phi_current / 1.5)
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.6 * np.exp(-err_phi_current / 1.5)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -20 * np.exp(-koz_margin_min * 6.0) * max(ang_vel_norm, 0.05)
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -20.0 * (1.0 - koz_margin_min/0.17) * max(ang_vel_norm, 0.05)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod99":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: idea: penalty based
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0:
                r1 = err_phi_delta * (20 * np.exp(-err_phi_current * 0.2) + 0.05)
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 1.5))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 1.5))
            r2 += -0.05 * err_phi_current / 180.0
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            for margin in koz_margins_violated:
                # Maximum penalty inside of KOZ
                r5 += -20 * np.exp(-margin * 6.0) * max(ang_vel_norm, 0.05)
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            if 0 < koz_margin_min < 0.17:
                r5 = -5 * (1.0 - koz_margin_min/0.17) * max(ang_vel_norm, 0.05)

        r_speed = 0.0
        if ang_vel_norm > 0.0874 and koz_margin_min < 0.26: # around 5 deg/s
            r_speed = 60 * (0.0874-ang_vel_norm) * (0.26-koz_margin_min)
        elif ang_vel_norm > 0.17:
            r_speed = -1.0

        r_koz_direction = 0.0
        if koz_margin_delta < 0 and koz_margin_min > 0:
            r_koz_direction = 200* koz_margin_delta * np.exp(-koz_margin_min * 20.0)

        r_total = r1 + r2 + r4 + r5 + r_speed + r_koz_direction

    if USE_REWARD == "mod991":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: idea: penalty based
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0:
                r1 = 2 * err_phi_delta * (20 * np.exp(-err_phi_current * 0.2) + 1)
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 1.5))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 1.5))
            r2 += -0.05 * err_phi_current / 180.0
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            for margin in koz_margins_violated:
                # Maximum penalty inside of KOZ
                r5 += -80 * np.exp(-margin * 6.0) * max(ang_vel_norm, 0.05)
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            if 0 < koz_margin_min < 0.17:
                r5 = -20 * (1.0 - koz_margin_min/0.17) * max(ang_vel_norm, 0.05)

        r_speed = 0.0
        if ang_vel_norm > 0.0874 and koz_margin_min < 0.26: # around 5 deg/s
            r_speed = 60 * (0.0874-ang_vel_norm) * (0.26-koz_margin_min)
        elif ang_vel_norm > 0.17:
            r_speed = -1.0

        r_koz_direction = 0.0
        if koz_margin_delta < 0 and koz_margin_min > 0:
            r_koz_direction = 300* koz_margin_delta * np.exp(-koz_margin_min * 20.0)

        print(f"[{time_elapsed:.1f}] r1={r1:.3f}, r2={r2:.3f}, r4={r4:.3f}, r5={r5:.3f}, r_speed={r_speed:.3f}, r_koz_dir={r_koz_direction:.3f}")

        r_total = r1 + r2 + r4 + r5 + r_speed + r_koz_direction

    if USE_REWARD == "mod224ph2c2a":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r2 = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.08: # around 4.5 deg/s
            r6 = -1.0

        
        r_total = r1 + r2 + r4 + r5 + r6

    if USE_REWARD == "mod224ph2c2b":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty, different weight for pointing reward
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 20.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if koz_margin_min > 0.17:
                r2 = 0.2 * np.exp(-(err_phi_current / 20.0))

                if err_phi_current < 1.0:
                    r2 += 0.1
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.08: # around 4.5 deg/s
            r6 = -1.0

        
        r_total = r1 + r2 + r4 + r5 + r6

    if USE_REWARD == "mod224ph2c2c":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty, different weight for pointing reward, small constant penalty while not within 1° acc.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 20.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if koz_margin_min > 0.17:
                r2 = 0.2 * np.exp(-(err_phi_current / 20.0))

                if err_phi_current <= 1.0:
                    r2 += 0.1

            if err_phi_current > 1.0:
                r2 -= 0.05
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.08: # around 4.5 deg/s
            r6 = -1.0

        
        r_total = r1 + r2 + r4 + r5 + r6

    if USE_REWARD == "mod224ph2c2d":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty, different weight for pointing reward, small constant penalty while not within 1° acc, no penalty for moving away from target if moving away from KOZ and being close to KOZ
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 2*0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0.17:
                # If KOZ margin between [0.17, 2*0.17] and moving away from KOZ, dont penalize moving away from target.
                if err_phi_delta < 0 and koz_margin_delta > 0:
                    r1 = 0
                else:
                    r1 = err_phi_delta
            elif koz_margin_min > 0:
                # If KOZ margin between [0, 0.17] and moving away from KOZ, dont penalize moving away from target.
                if err_phi_delta < 0 and koz_margin_delta > 0:
                    r1 = 0
                else:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 20.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if koz_margin_min > 0.17:
                r2 = 0.2 * np.exp(-(err_phi_current / 20.0))

                if err_phi_current <= 1.0:
                    r2 += 0.1

            if err_phi_current > 1.0:
                r2 -= 0.05
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.08: # around 4.5 deg/s
            r6 = -1.0

        
        r_total = r1 + r2 + r4 + r5 + r6

    if USE_REWARD == "mod224ph2c2e":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty, different weight for pointing reward, small constant penalty while not within 1° acc, reward for moving tangential to KOZ,
                no penalty for moving away from target if within a buffer zone round KOZ.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 2*0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0.17:
                if err_phi_delta > 0:
                    r1 = err_phi_delta
                else:
                    r1 = 0
            elif koz_margin_min > 0:
                if err_phi_delta > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.2 * np.exp(-(err_phi_current / 20.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if koz_margin_min > 0.17:
                r2 = 0.2 * np.exp(-(err_phi_current / 20.0))

                if err_phi_current <= 1.0:
                    r2 += 0.1

            if err_phi_current > 1.0:
                r2 -= 0.05
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.08: # around 4.5 deg/s
            r6 = -1.0

        # Bonus for moving tangential around KOZ.
        r7 = 0.0
        # If koz_margin_min in [0.17, 2*0.17]
        if koz_margin_min >= 0.17 and koz_margin_min <= 2*0.17:
            r7 = ang_vel_norm * 180.0 / np.pi * 0.05

        
        r_total = r1 + r2 + r4 + r5 + r6 + r7

    if USE_REWARD == "mod224ph2c2f":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty, different weight for pointing reward, small constant penalty while not within 1° acc, reward for moving tangential to KOZ,
                no penalty for moving away from target if within a buffer zone round KOZ, prevent moving back and forth around KOZ, adjust weights. 
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = 1 * err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 2*0.17:
                r1 = 1 * err_phi_delta
            elif koz_margin_min > 0.17:
                r1 = 1 * err_phi_delta
            elif koz_margin_min > 0:
                if err_phi_delta > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.4 * np.exp(-(err_phi_current / 15.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if koz_margin_min > 0.17:
                r2 = 0.4 * np.exp(-(err_phi_current / 15.0))

                if err_phi_current <= 1.0:
                    r2 += 0.1

            # No bonus within 0.17 KOZ safety margin
            else:
                r2 = 0.0

            if err_phi_current > 1.0:
                r2 -= 0.1
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))
        r4 -= 100.0 * np.sqrt((torque_1-torque_1_prev)**2 + (torque_2-torque_2_prev)**2 + (torque_3-torque_3_prev)**2)

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -2.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.08: # around 4.5 deg/s
            r6 = -1.0

        # Bonus for moving tangential around KOZ.
        r7 = 0.0
        # If koz_margin_min in [0.17, 2*0.17]
        if koz_margin_min >= 0.17 and koz_margin_min <= 2*0.17:
            r7 = 4.0 * (ang_vel_norm * 180.0 / np.pi * 0.05) * np.exp(- abs(koz_margin_delta) * 1000.0)

        print(f"[{time_elapsed:.1f}] r1={r1:.3f}, r2={r2:.3f}, r4={r4:.3f}, r5={r5:.3f}, r6={r6:.3f}, r7={r7:.3f} | vel={ang_vel_norm:.3f}, delta_koz_margin_min={koz_margin_delta:.5f}")
        
        r_total = r1 + r2 + r4 + r5 + r6 + r7

    if USE_REWARD == "mod224ph2c2g":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty, different weight for pointing reward, small constant penalty while not within 1° acc, reward for moving tangential to KOZ,
                no penalty for moving away from target if within a buffer zone round KOZ, prevent moving back and forth around KOZ, adjust weights,
                fix exploit rotating around boresight vector. 
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = 1 * err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 2*0.17:
                r1 = 1 * err_phi_delta
            elif koz_margin_min > 0.17:
                r1 = 1 * err_phi_delta
            elif koz_margin_min > 0:
                if err_phi_delta > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.5 * np.exp(-(err_phi_current / 25.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if koz_margin_min > 0.17:
                r2 = 0.5 * np.exp(-(err_phi_current / 25.0))

                if err_phi_current <= 1.0:
                    r2 += 0.4

            # No bonus within 0.17 KOZ safety margin
            else:
                r2 = 0.0

            if err_phi_current > 1.0:
                r2 -= 0.1
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))
        r4 -= 20.0 * np.sqrt((torque_1-torque_1_prev)**2 + (torque_2-torque_2_prev)**2 + (torque_3-torque_3_prev)**2)

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -2.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.12: # around 6.9 deg/s
            r6 = -1.0

        # Bonus for moving tangential around KOZ. Need to use the velocity vector which excludes the component along the boresight vector. This is to prevent exploiting the reward by rotating around the boresight vector.
        r7 = 0.0
        ang_vel_eff_norm = np.sqrt(ang_vel_sat_y**2 + ang_vel_sat_z**2) # exclude x component as it is boresight vector.
        
        # If koz_margin_min in [0.17, 2*0.17]
        if koz_margin_min >= 0.17 and koz_margin_min <= 2*0.17:
            r7 = 3.0 * (ang_vel_eff_norm * 180.0 / np.pi * 0.05) * np.exp(- abs(koz_margin_delta) * 1000.0)

        print(f"[{time_elapsed:.1f}] r1={r1:.3f}, r2={r2:.3f}, r4={r4:.3f}, r5={r5:.3f}, r6={r6:.3f}, r7={r7:.3f} | vel={ang_vel_norm:.3f}, vel_b={ang_vel_eff_norm:.3f}, delta_koz_margin_min={koz_margin_delta:.5f}")
        
        r_total = r1 + r2 + r4 + r5 + r6 + r7

    if USE_REWARD == "mod224ph2c2h":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: ang velocity penalty, different weight for pointing reward, small constant penalty while not within 1° acc, reward for moving tangential to KOZ,
                no penalty for moving away from target if within a buffer zone round KOZ, prevent moving back and forth around KOZ, adjust weights,
                fix exploit rotating around boresight vector. 
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = 1 * err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 2*0.17:
                r1 = 1 * err_phi_delta * np.exp(-err_phi_current / 45.0)
            elif koz_margin_min > 0.17:
                r1 = 1 * err_phi_delta * np.exp(-err_phi_current / 45.0)
            elif koz_margin_min > 0:
                if err_phi_delta > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17) * np.exp(-err_phi_current / 45.0)
                else:
                    r1 = 0
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r2 = 0.5 * np.exp(-(err_phi_current / 25.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if koz_margin_min > 0.17:
                r2 = 0.5 * np.exp(-(err_phi_current / 25.0))

                if err_phi_current <= 1.0:
                    r2 += 0.4

            # No bonus within 0.17 KOZ safety margin
            else:
                r2 = 0.0

            if err_phi_current > 1.0:
                r2 -= 0.1
            

        # Penalty for using large torques
        r4 = - 10.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))
        r4 -= 20.0 * np.sqrt((torque_1-torque_1_prev)**2 + (torque_2-torque_2_prev)**2 + (torque_3-torque_3_prev)**2)

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r5 = -2.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        r6 = 0.0
        if ang_vel_norm > 0.12: # around 6.9 deg/s
            r6 = -1.0

        # Bonus for moving tangential around KOZ. Need to use the velocity vector which excludes the component along the boresight vector. This is to prevent exploiting the reward by rotating around the boresight vector.
        r7 = 0.0
        ang_vel_eff_norm = np.sqrt(ang_vel_sat_y**2 + ang_vel_sat_z**2) # exclude x component as it is boresight vector.
        
        # If koz_margin_min in [0.17, 2*0.17]
        if koz_margin_min >= 0.17 and koz_margin_min <= 2*0.17:
            r7 = 2.0 * (ang_vel_eff_norm * 180.0 / np.pi * 0.05) * np.exp(- abs(koz_margin_delta) * 1000.0)

        #print(f"[{time_elapsed:.1f}] r1={r1:.3f}, r2={r2:.3f}, r4={r4:.3f}, r5={r5:.3f}, r6={r6:.3f}, r7={r7:.3f} | vel={ang_vel_norm:.3f}, vel_b={ang_vel_eff_norm:.3f}, delta_koz_margin_min={koz_margin_delta:.5f}")
        
        r_total = r1 + r2 + r4 + r5 + r6 + r7

    if USE_REWARD == "mod224ph2c3":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: higher reward for desired pointing and close to target
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current <= 0.25:
                r2 = 0.4
            else:
                r2 = 0.1 * np.exp((-err_phi_current) * 1.0)
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if err_phi_current <= 0.25:
                r2 = 0.4
            else:
                r2 = 0.1 * np.exp((-err_phi_current) * 1.0)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod224ph2d":
        """
        Goal: use mod224 from phase 1 as baseline for phase 2 tuning
        Result: 
        Note: higher reward for desired pointing. different weight.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == "phase 1":
            r1 = err_phi_delta
            
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r1 = err_phi_delta
            elif koz_margin_min > 0:
                r1 = err_phi_delta * (koz_margin_min/0.17)
            else:
                r1 = 0
            

        # Bonus for high accuracy
        r2 = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            if err_phi_current <= 0.25:
                r2 = 0.3
            else:
                r2 = 0.1 * np.exp((-err_phi_current + 0.25) * 1.0)
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            if err_phi_current <= 0.25:
                r2 = 0.3
            else:
                r2 = 0.1 * np.exp((-err_phi_current + 0.25) * 1.0)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod23":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: Adding the -0.01 penalty makes both accuracy and violation worse.
        Note: Added -0.01 penalty when not within desired accuracy.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            else:
                r2 = -0.01
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = -0.1
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            else:
                r2 = -0.01
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod24":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: worse than before
        Note: Increased bonus for desired pointing accuracy
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.1
            elif err_phi_current < 0.25:
                r2 = 0.1 * (((0.25-err_phi_current)/(10.0/9.0 * 0.05)) + 0.1)
            
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.1
            elif err_phi_current < 0.25:
                r2 = 0.1 * (((0.25-err_phi_current)/(10.0/9.0 * 0.05)) + 0.1)
            
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod25":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: Worse
        Note: Removed constant terms for reducing attitude error. Removed torque penalty. Adjusted pointing bonus.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.0 * err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.0 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.02 + 0.1*(0.25-err_phi_current) 
    
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.25:
                r2 = 0.02 + 0.1*(0.25-err_phi_current) 
            
        # Penalty for using large torques
        r4 = 0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod26":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: 
        Note: Removed constant terms for reducing attitude error. Removed torque penalty. Adjusted pointing bonus.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.0 * err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.0 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.04 + 0.1*(0.25-err_phi_current)
            # Bonus for being close to desired accuracy
            elif err_phi_current < 2.0:
                r2 = 0.04 * (1.0/(2.0-0.25) * (2.0-err_phi_current))**2
    
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.25:
                r2 = 0.04 + 0.1*(0.25-err_phi_current)
            # Bonus for being close to desired accuracy
            elif err_phi_current < 2.0:
                r2 = 0.04 * (1.0/(2.0-0.25) * (2.0-err_phi_current))**2
            
        # Penalty for using large torques
        r4 = 0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod27":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: higher accuracy bonus, s-shaped function.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.01 + 0.2*(np.cbrt(0.125) - np.cbrt(err_phi_current - 0.125)) # [0.01, 0.21]
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.25:
                r2 = 0.01 + 0.2*(np.cbrt(0.125) - np.cbrt(err_phi_current - 0.125))
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod28":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod29":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1. remove torque penalty.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = 0.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod30":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1. remove torque penalty. add back bad accuracy penalty
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # Penalty if violated KOZ or not within desired accuracy
            if koz_violation_cnt > 0 or err_phi_current >= 0.25:
                r2 = -0.04
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = 0.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod31":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1. remove torque penalty.
                change bad accuracy penalty to based on timestep
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # Penalty if violated KOZ or not within desired accuracy
            if koz_violation_cnt > 0 or err_phi_current >= 0.25:
                r2 = -0.000005 * time_elapsed**2 * err_phi_current
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = 0.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod32":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: increased accuracy bonus
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.1
            elif err_phi_current < 0.25:
                r2 = 0.1 * (((0.25-err_phi_current)/0.05)) + 0.01
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod33":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: removed pointing accuracy bonus not given if KOZ violation. added pointing accuracy penalty
                when too far away. increased KOZ penalty. added persistent penalty once inside KOZ.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.03
            elif err_phi_current < 0.25:
                r2 = 0.03 * (0.25-err_phi_current) / 0.05 # [0, 0.03]
            else:
                r2 = -0.01 * min((err_phi_current-0.25)/4.75, 1.0) # [-0.01, 0]
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -2.0 * (1.0 - koz_margin_min/0.17)

            # Add persistent penalty for the remaining episode once inside KOZ
            if koz_violation_cnt > 0:
                r5 = r5 - 0.05

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod34":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: diff mod33: removed constant reward in r1. slightly increased accuracy penalty.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.03
            elif err_phi_current < 0.25:
                r2 = 0.03 * (0.25-err_phi_current) / 0.05 # [0, 0.03]
            else:
                r2 = -0.01 - 0.01 * min((err_phi_current-0.25)/4.75, 1.0) # [-0.02, -0.01]
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -2.0 * (1.0 - koz_margin_min/0.17)

            # Add persistent penalty for the remaining episode once inside KOZ
            if koz_violation_cnt > 0:
                r5 = r5 - 0.05

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod35":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: diff mod34: added settling reward
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if koz_margin_min > 0.17:
                    r1 = err_phi_delta
                elif koz_margin_min > 0:
                    r1 = err_phi_delta * (koz_margin_min/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.03
            elif err_phi_current < 0.25:
                r2 = 0.03 * (0.25-err_phi_current) / 0.05 # [0, 0.03]
            else:
                r2 = -0.01 - 0.01 * min((err_phi_current-0.25)/4.75, 1.0) # [-0.02, -0.01]
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -2.0 * (1.0 - koz_margin_min/0.17)

            # Add persistent penalty for the remaining episode once inside KOZ
            if koz_violation_cnt > 0:
                r5 = r5 - 0.05

        # Reward if settled
        r6 = 0.0
        if phase == 2:
            if err_phi_current < 0.25 and ang_vel_norm < 0.001:
                r6 = 0.02
        
        r_total = r1 + r2 + r4 + r5 + r6

    if USE_REWARD == "mod37":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: take a step back, see what causes pointing accuracy to become worse.
        """

        # Reward for reducing attitude error
        r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.2:
            r2 = 0.02
        elif err_phi_current < 0.25:
            r2 = 0.02 * ((0.25-err_phi_current)/0.05)
        # Penalty if farther away
        else:
            r2 = -0.01


        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -1.0
            # No penalty if farther away
            else:
                r5 = 0.0

        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod38":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: take a step back, see what causes pointing accuracy to become worse. Less KOZ penalty
        """

        # Reward for reducing attitude error
        r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.2:
            r2 = 0.2
        elif err_phi_current < 0.25:
            r2 = 0.2 * ((0.25-err_phi_current)/0.05)
        # Penalty if farther away
        else:
            r2 = -0.01


        # Penalty for using large torques
        r4 = - 0.01*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r5 = -0.01
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min < 0.17:
                r5 = -0.01*math.exp(-60.0*koz_margin_min)
            # No penalty if farther away
            else:
                r5 = 0.0

        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod39":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: see if obs change from (0,0,0,0) in ph1 to (a,b,c,d) in ph2 is causing issues regardless of reward shape.
        """

        # Reward for reducing attitude error
        r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.2:
            r2 = 0.02
        elif err_phi_current < 0.25:
            r2 = 0.02 * ((0.25-err_phi_current)/0.05)
        # Penalty if farther away
        else:
            r2 = -0.01


        # Penalty for using large torques
        r4 = -1*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0

        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod41":

        # Reward for reducing attitude error
        r1 = min(max(0.1 * err_phi_delta,-0.02), 0.02) # [-0.02, 0.02]

        # Pointing accuracy
        r2 = - (err_phi_current / 180.0 + min(err_phi_current / 5.0, 1.0)**2) # [-2, 0]

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == "phase 2":
            r5 = - min(max((0.17 - koz_margin_min) / 0.17, 0.0), 1.0)**2 # [-1, 0]
      
        r_total = r1 + 0.01*r2 + 0.01*r5

        # Termination reward for entering KOZ
        if termination == "failure":
            r_total = -250.0

        # Termination reward for settling for a certain duration.
        elif termination == "success":
            r_total = 150.0

    return r_total