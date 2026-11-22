# run_hls_v0.tcl - Vitis HLS: v0 (port directo)
# Uso: vitis_hls -f scripts/run_hls_v0.tcl   (desde "Prototipo en FPGA/")
open_project -reset proj_lk_v0
set_top lk_hls_v0
add_files hls/v0_port_directo/lk_hls_v0.cpp -cflags "-Ihls/include"
add_files -tb tb/tb_lk_hls.cpp -cflags "-Ihls/include -DLK_TB_V0"
open_solution -reset sol1 -flow_target vitis
set_part xck26-sfvc784-2LV-c
create_clock -period 5 -name default
csim_design
csynth_design
cosim_design
export_design -format xo -output ../results/fpga/lk_hls_v0.xo
exit
