"""Test regions = the simulated master's hidden rules (only for evaluation)."""
REGIONS = {
 "R1 lip 2-4mm single":     lambda t: t["zone"]=="sealing_lip" and 2.0<=t["size_mm"]<=4.0 and t["count"]==1,
 "R2 lip 2 pits <2mm":      lambda t: t["zone"]=="sealing_lip" and t["size_mm"]<2.0 and t["count"]==2,
 "R3 flange round 4-5.5":   lambda t: t["zone"]=="flange_face" and t["shape"]=="round" and 4.0<t["size_mm"]<=5.5 and t["count"]==1,
 "R4 flange irreg 3-4":     lambda t: t["zone"]=="flange_face" and t["shape"]=="irregular" and 3.0<t["size_mm"]<=4.0 and t["count"]<3,
 "R5 rim 4-6":              lambda t: t["zone"]=="rim" and 4.0<t["size_mm"]<=6.0 and t["count"]<3,
}
