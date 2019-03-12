#include "pose.h"

/**
 * Function list: 
 * cin_load
 * weight_load
 * inter_load
 * depth_conv
 * conv
 * relu
 * maxpool
 * upsample - reserved
 * inter_write
 * cout_write
 * engine
 */

/*
 * Function name: cin_load_ddr_read
 * Function description: This function loads cin results from off-chip DRAM.
 *                       Two modes are enabled. If the whole feature maps of the layer could fit
 *                       in the on-chip buffer, they will be loaded as a whole. Otherwise, each time,
 *                       LAYER_IN_NUM_T * (LAYER_IN_W_T + FILTER_S - 1) of data are loaded.
 */
void cin_load_ddr_read(
  bus_t0  *global_cin,
  bus_t0  cin_burst_buf[],
  uint    LAYER_IN_H_HW,
  uint    LAYER_IN_W_HW,
  uint    LAYER_IN_NUM_T,
  uint    LAYER_IN_H_T,
  uint    LAYER_IN_W_T,
  uint    FILTER_S,
  uint    cin_offset,
  uint    in_num_iter,
  uint    in_h_iter,
  uint    in_w_iter,
  bool    max_pool
){
  if ((LAYER_IN_H_HW <= IN_H_T + K_T - 1) && (LAYER_IN_W_HW <= IN_W_T + K_T - 1) && !max_pool){
  // Load the patch as a whole if it fits in on-chip buffer.
    uint global_cin_offset = in_num_iter * LAYER_IN_H_HW * LAYER_IN_W_HW + cin_offset;
    memcpy((void*)cin_burst_buf, (void*)&global_cin[global_cin_offset / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_NUM_T * LAYER_IN_H_HW * LAYER_IN_W_HW);
  } else {
    for (int hh = 0; hh < LAYER_IN_H_T + FILTER_S - 1; hh++){
      uint h = in_h_iter + hh;
      uint local_cin_offset = hh * (LAYER_IN_W_T + FILTER_S - 1) * LAYER_IN_NUM_T;
      uint global_cin_offset = in_num_iter * LAYER_IN_H_HW * LAYER_IN_W_HW + h * LAYER_IN_W_HW * LAYER_IN_NUM_T + in_w_iter * LAYER_IN_NUM_T + cin_offset;
      memcpy((void*)&cin_burst_buf[local_cin_offset / BUS_PACK_FACTOR0], (void*)&global_cin[global_cin_offset / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_NUM_T * (LAYER_IN_W_T + FILTER_S - 1));
    }
  }
}

/*
 * Function name: cin_load_fifo_write
 * Function description: This function writes cin data to the downstream modules.
 */
void cin_load_fifo_write(
  bus_t0                         cin_burst_buf[],
  hls::stream<CinLoadData0Type>  &fifo_cin,
  uint                           LAYER_IN_NUM_T,
  uint                           LAYER_IN_H_T,
  uint                           LAYER_IN_W_T,
  uint                           FILTER_S
){
  int ii = 0;
  int hh = 0;
  int ww = 0;
  bool done = 0;
  while(!done){
#pragma HLS PIPELINE II=1
    uint local_cin_idx = hh * (LAYER_IN_W_T + FILTER_S - 1) * LAYER_IN_NUM_T + ww * LAYER_IN_NUM_T + ii * DEPTH_CONV_LANE;
    uint bus_cin_idx = local_cin_idx / BUS_PACK_FACTOR0;
    uint bus_cin_offset = local_cin_idx % BUS_PACK_FACTOR0;
    bus_t0 bus_cin_data = cin_burst_buf[bus_cin_idx];
    CinLoadData0Type fifo_cin_data;
#if DATA_SEL_FACTOR0 == 1
    fifo_cin_data = bus_cin_data;
#elif DATA_SEL_FACTOR0 == 2 
    switch(bus_cin_offset / DEPTH_CONV_LANE){
      case 0:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
        break;
      case 1:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
        break;
    }         
#elif DATA_SEL_FACTOR0 == 4
    switch(bus_cin_offset / DEPTH_CONV_LANE){
      case 0:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
        break;
      case 1:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
        break;
      case 2:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 3 - 1, DATA_W0 * DEPTH_CONV_LANE * 2);
        break;
      case 3:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 4 - 1, DATA_W0 * DEPTH_CONV_LANE * 3);
        break;
    }           
#elif DATA_SEL_FACTOR0 == 8
    switch(bus_cin_offset / DEPTH_CONV_LANE){
      case 0:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
        break;
      case 1:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
        break;
      case 2:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 3 - 1, DATA_W0 * DEPTH_CONV_LANE * 2);
        break;
      case 3:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 4 - 1, DATA_W0 * DEPTH_CONV_LANE * 3);
        break;
      case 4:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 5 - 1, DATA_W0 * DEPTH_CONV_LANE * 4);
        break;
      case 5:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 6 - 1, DATA_W0 * DEPTH_CONV_LANE * 5);
        break;
      case 6:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 7 - 1, DATA_W0 * DEPTH_CONV_LANE * 6);
        break;
      case 7:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 8 - 1, DATA_W0 * DEPTH_CONV_LANE * 7);
        break;
    }           
#elif DATA_SEL_FACTOR0 == 16
    switch(bus_cin_offset / DEPTH_CONV_LANE){
      case 0:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
        break;
      case 1:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
        break;
      case 2:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 3 - 1, DATA_W0 * DEPTH_CONV_LANE * 2);
        break;
      case 3:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 4 - 1, DATA_W0 * DEPTH_CONV_LANE * 3);
        break;
      case 4:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 5 - 1, DATA_W0 * DEPTH_CONV_LANE * 4);
        break;
      case 5:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 6 - 1, DATA_W0 * DEPTH_CONV_LANE * 5);
        break;
      case 6:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 7 - 1, DATA_W0 * DEPTH_CONV_LANE * 6);
        break;
      case 7:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 8 - 1, DATA_W0 * DEPTH_CONV_LANE * 7);
        break;
      case 8:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 9 - 1, DATA_W0 * DEPTH_CONV_LANE * 8);
        break;
      case 9:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 10 - 1, DATA_W0 * DEPTH_CONV_LANE * 9);
        break;
      case 10:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 11 - 1, DATA_W0 * DEPTH_CONV_LANE * 10);
        break;
      case 11:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 12 - 1, DATA_W0 * DEPTH_CONV_LANE * 11);
        break;
      case 12:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 13 - 1, DATA_W0 * DEPTH_CONV_LANE * 12);
        break;
      case 13:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 14 - 1, DATA_W0 * DEPTH_CONV_LANE * 13);
        break;
      case 14:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 15 - 1, DATA_W0 * DEPTH_CONV_LANE * 14);
        break;
      case 15:
        fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 16 - 1, DATA_W0 * DEPTH_CONV_LANE * 15);
        break;
    }           
#endif            
    fifo_cin.write(fifo_cin_data);

    ww++;
    if (ww == LAYER_IN_W_T + FILTER_S - 1){
      ww = 0;
      hh++;
      if (hh == LAYER_IN_H_T + FILTER_S - 1){
        hh = 0;
        ii++;
        if (ii == LAYER_IN_NUM_T / DEPTH_CONV_LANE){
          ii = 0;
          done = 1;
        }
      }
    }
  }
}

/*
 * Function name: cin_load
 * Function description: This function loads and distributes cin and instructions.
 */
void cin_load(
  bus_t0                         *global_cin, 
  uint                           config[CONFIG_PARAMS],
  hls::stream<CinLoadData0Type>  &fifo_cin,
  hls::stream<ConfigInst>        &fifo_config_out
){
#pragma HLS INLINE off 
  // on-chip buffer for cin data
  bus_t0 cin_burst_buf_ping[IN_NUM_T * (IN_H_T + K_T - 1) * (IN_W_T + K_T - 1) / BUS_PACK_FACTOR0];
  bus_t0 cin_burst_buf_pong[IN_NUM_T * (IN_H_T + K_T - 1) * (IN_W_T + K_T - 1) / BUS_PACK_FACTOR0];

  // layer batch
  ap_uint<32> LAYER_BATCH = config[24];
  
  // tiling iterators
  uint in_num_iter = 0;
  uint out_num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;

  // parameters
  // inst0
  ap_uint<32> LAYER_IN_NUM_HW; 
  ap_uint<32> LAYER_OUT_NUM_HW;
  ap_uint<32> LAYER_IN_H_HW;
  ap_uint<32> LAYER_IN_W_HW;  
  ap_uint<32> LAYER_OUT_H_HW;  
  ap_uint<32> LAYER_OUT_W_HW;
  // inst1
  ap_uint<32> LAYER_IN_NUM; 
  ap_uint<32> LAYER_OUT_NUM;
  ap_uint<32> LAYER_IN_H;
  ap_uint<32> LAYER_IN_W;  
  ap_uint<32> LAYER_OUT_H;  
  ap_uint<32> LAYER_OUT_W;  
  // inst2
  ap_uint<32> CIN_OFFSET; 
  ap_uint<32> WEIGHT_OFFSET;
  ap_uint<32> BIAS_OFFSET;
  ap_uint<32> COUT_OFFSET; 
  ap_uint<16> FILTER_S1; 
  ap_uint<16> FILTER_S2;   
  ap_uint<32> STRIDE;  
  // inst3
  ap_uint<32> LAYER_EN; 
  ap_uint<32> LAYER_IN_NUM_T;
  ap_uint<32> LAYER_OUT_NUM_T;
  ap_uint<32> LAYER_IN_H_T;
  ap_uint<32> LAYER_IN_W_T;
  // inst4
  ap_uint<32> LAYER_TASK_NUM1;
  ap_uint<32> LAYER_TASK_NUM2;
  ap_uint<32> LAYER_LOCAL_ACCUM_NUM;
  ap_uint<32> LAYER_LOCAL_REG_NUM;
  ap_uint<32> LAYER_ROW_IL_FACTOR;
  ap_uint<32> LAYER_COL_IL_FACTOR;
    
  uint LAYER_IN_NUM_T_prev;
  uint LAYER_IN_H_T_prev;
  uint LAYER_IN_W_T_prev;
  uint FILTER_S_prev;

  uint task_cnt = 0;
  bool layer_start = 1;
	bool done = 0;
	// We assum that cin has been pre-padded with zeros
	while(!done){

    // inst0
    LAYER_IN_NUM_HW  = config[0 + layer_iter * CONFIG_PARAMS];
    LAYER_OUT_NUM_HW = config[1 + layer_iter * CONFIG_PARAMS];
    LAYER_IN_H_HW    = config[2 + layer_iter * CONFIG_PARAMS];
    LAYER_IN_W_HW    = config[3 + layer_iter * CONFIG_PARAMS];
    LAYER_OUT_H_HW   = config[4 + layer_iter * CONFIG_PARAMS];
    LAYER_OUT_W_HW   = config[5 + layer_iter * CONFIG_PARAMS];
  
    // inst1
    LAYER_IN_NUM  = config[6 + layer_iter * CONFIG_PARAMS];
    LAYER_OUT_NUM = config[7 + layer_iter * CONFIG_PARAMS];
    LAYER_IN_H    = config[8 + layer_iter * CONFIG_PARAMS];
    LAYER_IN_W    = config[9 + layer_iter * CONFIG_PARAMS];
    LAYER_OUT_H   = config[10 + layer_iter * CONFIG_PARAMS];
    LAYER_OUT_W   = config[11 + layer_iter * CONFIG_PARAMS];
   
    // inst2
    CIN_OFFSET    = config[12 + layer_iter * CONFIG_PARAMS];
    WEIGHT_OFFSET = config[13 + layer_iter * CONFIG_PARAMS];
    BIAS_OFFSET   = config[14 + layer_iter * CONFIG_PARAMS];
    COUT_OFFSET   = config[15 + layer_iter * CONFIG_PARAMS];
    FILTER_S1     = config[16 + layer_iter * CONFIG_PARAMS];
    FILTER_S2     = config[17 + layer_iter * CONFIG_PARAMS];
    STRIDE        = config[18 + layer_iter * CONFIG_PARAMS];
  
    // inst3
    LAYER_EN        = config[19 + layer_iter * CONFIG_PARAMS];
    LAYER_IN_NUM_T  = config[20 + layer_iter * CONFIG_PARAMS];
    LAYER_OUT_NUM_T = config[21 + layer_iter * CONFIG_PARAMS];
    LAYER_IN_H_T    = config[22 + layer_iter * CONFIG_PARAMS];
    LAYER_IN_W_T    = config[23 + layer_iter * CONFIG_PARAMS];
  
    ap_uint<1>  DEPTH_CONV_EN  = LAYER_EN[0];
    ap_uint<1>  CONV_EN        = LAYER_EN[1];
    ap_uint<1>  RELU_EN        = LAYER_EN[2];
    ap_uint<1>  POOL_EN        = LAYER_EN[3];
    ap_uint<1>  UP_SAMPLE_EN   = LAYER_EN[4];  // reserved
    ap_uint<1>  BIAS_EN        = LAYER_EN[5];
    ap_uint<1>  INTER_LOAD_EN  = LAYER_EN[6];  
    ap_uint<1>  INTER_WRITE_EN = LAYER_EN[7]; 
 
    // inst4
    LAYER_TASK_NUM1       = config[25 + layer_iter * CONFIG_PARAMS];
    LAYER_TASK_NUM2       = config[26 + layer_iter * CONFIG_PARAMS];
    LAYER_LOCAL_ACCUM_NUM = config[27 + layer_iter * CONFIG_PARAMS];
    LAYER_LOCAL_REG_NUM   = config[28 + layer_iter * CONFIG_PARAMS];
    LAYER_ROW_IL_FACTOR   = config[29 + layer_iter * CONFIG_PARAMS];
    LAYER_COL_IL_FACTOR   = config[30 + layer_iter * CONFIG_PARAMS];

    ConfigInst inst0 = (LAYER_OUT_W_HW, LAYER_OUT_H_HW, LAYER_IN_W_HW, LAYER_IN_H_HW, LAYER_OUT_NUM_HW, LAYER_IN_NUM_HW);
    ConfigInst inst1 = (LAYER_OUT_W, LAYER_OUT_H, LAYER_IN_W, LAYER_IN_H, LAYER_OUT_NUM, LAYER_IN_NUM);
    ConfigInst inst2 = (STRIDE, FILTER_S2, FILTER_S1, COUT_OFFSET, BIAS_OFFSET, WEIGHT_OFFSET, CIN_OFFSET);
    ConfigInst inst3 = (LAYER_BATCH, LAYER_IN_W_T, LAYER_IN_H_T, LAYER_OUT_NUM_T, LAYER_IN_NUM_T, LAYER_EN);
    ConfigInst inst4 = (LAYER_COL_IL_FACTOR, LAYER_ROW_IL_FACTOR, LAYER_LOCAL_REG_NUM, LAYER_LOCAL_ACCUM_NUM, LAYER_TASK_NUM2, LAYER_TASK_NUM1);

    if (layer_start){
      fifo_config_out.write(inst0);
      fifo_config_out.write(inst1);
      fifo_config_out.write(inst2);
      fifo_config_out.write(inst3);
      fifo_config_out.write(inst4);
      layer_start = 0;
    } 
  
    // offsets
  	uint cin_offset = CIN_OFFSET;

    // set up some configuration signals
    uint FILTER_S = (DEPTH_CONV_EN == 1)? (uint)FILTER_S1: ((CONV_EN == 1)? (uint)FILTER_S2: 1);
    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);

    // execution
    if (INTER_LOAD_EN == 0){
      if ((max_pool && out_num_iter == 0) || separable_conv || conv2d){
        if (task_cnt == 0){
          cin_load_ddr_read(global_cin, cin_burst_buf_ping, LAYER_IN_H_HW, LAYER_IN_W_HW, LAYER_IN_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, FILTER_S, cin_offset, in_num_iter, in_h_iter, in_w_iter, max_pool);
        } else {
          if (task_cnt % 2 == 1){
            cin_load_ddr_read(global_cin, cin_burst_buf_pong, LAYER_IN_H_HW, LAYER_IN_W_HW, LAYER_IN_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, FILTER_S, cin_offset, in_num_iter, in_h_iter, in_w_iter, max_pool);
            cin_load_fifo_write(cin_burst_buf_ping, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev);  
          } else {
            cin_load_ddr_read(global_cin, cin_burst_buf_ping, LAYER_IN_H_HW, LAYER_IN_W_HW, LAYER_IN_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, FILTER_S, cin_offset, in_num_iter, in_h_iter, in_w_iter, max_pool);
            cin_load_fifo_write(cin_burst_buf_pong, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev);  
          }
        }
        task_cnt++;
        LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
        LAYER_IN_H_T_prev = LAYER_IN_H_T;
        LAYER_IN_W_T_prev = LAYER_IN_W_T;
        FILTER_S_prev = FILTER_S;
      }
    }

    in_num_iter += LAYER_IN_NUM_T;
    if (in_num_iter >= LAYER_IN_NUM){
      in_num_iter = 0;
      in_h_iter += LAYER_IN_H_T;      
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          out_num_iter += LAYER_OUT_NUM_T;
          if (out_num_iter >= LAYER_OUT_NUM){
            out_num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }        
          }
        }
      }
    }
  }

  if (task_cnt % 2 == 1){
    cin_load_fifo_write(cin_burst_buf_ping, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev);  
  } else {
    cin_load_fifo_write(cin_burst_buf_pong, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev); 
  }
}

/*
 * Function name: weight_load_depth_conv_weight_write
 * Function description: This function writes depth_conv weights to depth_conv module.
 */
void weight_load_depth_conv_weight_write(
  bus_t1 weight_burst_buf1[],
  hls::stream<WeightLoadData0Type> &fifo_depth_conv_weight,
  ConfigInst inst0,
  ConfigInst inst1,
  ConfigInst inst2,
  ConfigInst inst3,
  uint in_num_iter,
  uint out_num_iter
){
  // inst0
  ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
  // inst1
  ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
  // inst2
  ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
  ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
  ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
  ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
  ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
  ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
  ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
  // inst3
  ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
  ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
  ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
  ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

  ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
  ap_uint<1>  CONV_EN          = LAYER_EN[1];
  ap_uint<1>  RELU_EN          = LAYER_EN[2];
  ap_uint<1>  POOL_EN          = LAYER_EN[3];
  ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
 
  if (DEPTH_CONV_EN == 1){
    // write to downstream modules: depth_conv     
    int ii = 0;
    int p = 0;
    int q = 0;
    bool done = 0;
    while(!done){
#pragma HLS PIPELINE II=1
      uint local_w_idx = p * FILTER_S1 * LAYER_IN_NUM_T + q * LAYER_IN_NUM_T + ii * SIMD_LANE;
      uint bus_w_idx = local_w_idx / BUS_PACK_FACTOR1;
      uint bus_w_offset = local_w_idx % BUS_PACK_FACTOR1;
      bus_t1 bus_w_data = weight_burst_buf1[bus_w_idx];
      WeightLoadData0Type fifo_w_data;
#if DATA_SEL_FACTOR1 == 1
      fifo_w_data = bus_w_data;
#elif DATA_SEL_FACTOR1 == 2
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
      }
#elif DATA_SEL_FACTOR1 == 4         
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
        case 2:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
          break;
        case 3:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
          break;
      } 
#elif DATA_SEL_FACTOR1 == 8
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
        case 2:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
          break;
        case 3:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
          break;
        case 4:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
          break;
        case 5:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
          break;
        case 6:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
          break;
        case 7:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
          break;
      } 
#elif DATA_SEL_FACTOR1 == 16
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
        case 2:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
          break;
        case 3:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
          break;
        case 4:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
          break;
        case 5:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
          break;
        case 6:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
          break;
        case 7:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
          break;
        case 8:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 9 - 1, DATA_W1 * SIMD_LANE * 8);
          break;
        case 9:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 10 - 1, DATA_W1 * SIMD_LANE * 9);
          break;
        case 10:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 11 - 1, DATA_W1 * SIMD_LANE * 10);
          break;
        case 11:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 12 - 1, DATA_W1 * SIMD_LANE * 11);
          break;
        case 12:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 13 - 1, DATA_W1 * SIMD_LANE * 12);
          break;
        case 13:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 14 - 1, DATA_W1 * SIMD_LANE * 13);
          break;
        case 14:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 15 - 1, DATA_W1 * SIMD_LANE * 14);
          break;
        case 15:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 16 - 1, DATA_W1 * SIMD_LANE * 15);
          break;
      }           
#endif         
      fifo_depth_conv_weight.write(fifo_w_data);
      q++;
      if (q == FILTER_S1){
        q = 0;
        p++;
        if (p == FILTER_S1){
          p = 0;
          ii++;
          if (ii == LAYER_IN_NUM_T / SIMD_LANE){
            ii = 0;
            done = 1;
          }
        }
      }

    }
  }
}

/*
 * function name: weight_load_conv_weight_write
 * function description: this function writes conv weights to conv module.
 */
void weight_load_conv_weight_write(
  bus_t1 weight_burst_buf2[],
  hls::stream<WeightLoadData1Type> &fifo_conv_weight,
  ConfigInst inst0,
  ConfigInst inst1,
  ConfigInst inst2,
  ConfigInst inst3,
  uint in_num_iter,
  uint out_num_iter
){
  // inst0
  ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
  // inst1
  ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
  // inst2
  ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
  ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
  ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
  ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
  ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
  ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
  ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
  // inst3
  ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
  ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
  ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
  ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

  ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
  ap_uint<1>  CONV_EN          = LAYER_EN[1];
  ap_uint<1>  RELU_EN          = LAYER_EN[2];
  ap_uint<1>  POOL_EN          = LAYER_EN[3];
  ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved

  if (CONV_EN == 1){
//  	for (int oo = 0; oo < LAYER_OUT_NUM_T; oo++){
//      for (int p = 0; p < FILTER_S2; p++)
//        for (int q = 0; q < FILTER_S2; q++)
//          weight_write_loop2: for (int ii = 0; ii < LAYER_IN_NUM_T / SIMD_LANE; ii++){
    int oo = 0;
    int p = 0;
    int q = 0;
    int ii = 0;
    bool done = 0;
    while(!done){
#pragma HLS PIPELINE II=1            
      uint local_w_idx = oo * FILTER_S2 * FILTER_S2 * LAYER_IN_NUM_T + p * FILTER_S2 * LAYER_IN_NUM_T + q * LAYER_IN_NUM_T + ii * SIMD_LANE;
      uint bus_w_idx = local_w_idx / BUS_PACK_FACTOR1;
      uint bus_w_offset = local_w_idx % BUS_PACK_FACTOR1;
      bus_t1 bus_w_data = weight_burst_buf2[bus_w_idx];
      WeightLoadData1Type fifo_w_data;
#if DATA_SEL_FACTOR1 == 1
      fifo_w_data = bus_w_data;
#elif DATA_SEL_FACTOR1 == 2
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
      }
#elif DATA_SEL_FACTOR1 == 4
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
        case 2:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
          break;
        case 3:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
          break;
      }           
#elif DATA_SEL_FACTOR1 == 8
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
        case 2:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
          break;
        case 3:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
          break;
        case 4:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
          break;
        case 5:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
          break;
        case 6:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
          break;
        case 7:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
          break;
      }           
#elif DATA_SEL_FACTOR1 == 16
      switch(bus_w_offset / SIMD_LANE){
        case 0:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
          break;
        case 1:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
          break;
        case 2:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
          break;
        case 3:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
          break;
        case 4:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
          break;
        case 5:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
          break;
        case 6:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
          break;
        case 7:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
          break;
        case 8:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 9 - 1, DATA_W1 * SIMD_LANE * 8);
          break;
        case 9:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 10 - 1, DATA_W1 * SIMD_LANE * 9);
          break;
        case 10:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 11 - 1, DATA_W1 * SIMD_LANE * 10);
          break;
        case 11:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 12 - 1, DATA_W1 * SIMD_LANE * 11);
          break;
        case 12:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 13 - 1, DATA_W1 * SIMD_LANE * 12);
          break;
        case 13:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 14 - 1, DATA_W1 * SIMD_LANE * 13);
          break;
        case 14:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 15 - 1, DATA_W1 * SIMD_LANE * 14);
          break;
        case 15:
          fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 16 - 1, DATA_W1 * SIMD_LANE * 15);
          break;
      }           
#endif          
      fifo_conv_weight.write(fifo_w_data);

      ii++;
      if (ii == LAYER_IN_NUM_T / SIMD_LANE){
        ii = 0;
        q++;
        if (q == FILTER_S2){
          q = 0;
          p++;
          if (p == FILTER_S2){
            p = 0;
            oo++;
            if (oo == LAYER_OUT_NUM_T){
              oo = 0;
              done = 1;
            }
          }
        }
      }
    }
  }
}

/*
 * Function name: weight_load_bias_write
 * Function description: This function writes bias to relu module.
 */
void weight_load_bias_write(
  bus_t2 bias_burst_buf[],
  hls::stream<WeightLoadData2Type> &fifo_bias,
  ConfigInst inst0,
  ConfigInst inst1,
  ConfigInst inst2,
  ConfigInst inst3,
  uint in_num_iter,
  uint out_num_iter
){
  // inst0
  ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
  // inst1
  ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
  // inst2
  ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
  ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
  ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
  ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
  ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
  ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
  ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
  // inst3
  ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
  ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
  ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
  ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

  ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
  ap_uint<1>  CONV_EN          = LAYER_EN[1];
  ap_uint<1>  RELU_EN          = LAYER_EN[2];
  ap_uint<1>  POOL_EN          = LAYER_EN[3];
  ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved

  if (CONV_EN == 1){ 
    if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
      bias_write_loop: for (int oo = 0; oo < LAYER_OUT_NUM_T / SIMD_LANE; oo++){
#pragma HLS PIPELINE II=1
        uint local_b_idx = oo * SIMD_LANE;
        uint bus_b_idx = local_b_idx / BUS_PACK_FACTOR2;
        uint bus_b_offset = local_b_idx % BUS_PACK_FACTOR2;
        bus_t2 bus_b_data = bias_burst_buf[bus_b_idx];
        WeightLoadData2Type fifo_b_data;
#if DATA_SEL_FACTOR2 == 1
        fifo_b_data = bus_b_data;
#elif DATA_SEL_FACTOR2 == 2
        switch(bus_b_offset / SIMD_LANE){
          case 0:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
            break;
          case 1:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
            break;
        }
#elif DATA_SEL_FACTOR2 == 4
        switch(bus_b_offset / SIMD_LANE){
          case 0:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
            break;
          case 1:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
            break;
          case 2:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
            break;
          case 3:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
            break;
        }
#elif DATA_SEL_FACTOR2 == 8
        switch(bus_b_offset / SIMD_LANE){
          case 0:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
            break;
          case 1:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
            break;
          case 2:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
            break;
          case 3:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
            break;
          case 4:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 5 - 1, DATA_W2 * SIMD_LANE * 4);
            break;
          case 5:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 6 - 1, DATA_W2 * SIMD_LANE * 5);
            break;
          case 6:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 7 - 1, DATA_W2 * SIMD_LANE * 6);
            break;
          case 7:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 8 - 1, DATA_W2 * SIMD_LANE * 7);
            break;
        }
#elif DATA_SEL_FACTOR2 == 16
        switch(bus_b_offset / SIMD_LANE){
          case 0:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
            break;
          case 1:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
            break;
          case 2:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
            break;
          case 3:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
            break;
          case 4:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 5 - 1, DATA_W2 * SIMD_LANE * 4);
            break;
          case 5:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 6 - 1, DATA_W2 * SIMD_LANE * 5);
            break;
          case 6:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 7 - 1, DATA_W2 * SIMD_LANE * 6);
            break;
          case 7:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 8 - 1, DATA_W2 * SIMD_LANE * 7);
            break;
          case 8:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 9 - 1, DATA_W2 * SIMD_LANE * 8);
            break;
          case 9:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 10 - 1, DATA_W2 * SIMD_LANE * 9);
            break;
          case 10:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 11 - 1, DATA_W2 * SIMD_LANE * 10);
            break;
          case 11:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 12 - 1, DATA_W2 * SIMD_LANE * 11);
            break;
          case 12:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 13 - 1, DATA_W2 * SIMD_LANE * 12);
            break;
          case 13:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 14 - 1, DATA_W2 * SIMD_LANE * 13);
            break;
          case 14:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 15 - 1, DATA_W2 * SIMD_LANE * 14);
            break;
          case 15:
            fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 16 - 1, DATA_W2 * SIMD_LANE * 15);
            break;
        }
#endif       
        fifo_bias.write(fifo_b_data);
      }
    }
  }
}

/**
 * Function name: weight_load
 * Function description: This function loads weights and distributes them to downstream modules.
 */
void weight_load(
  bus_t1                           *global_weight, 
  bus_t2                           *global_bias,
  hls::stream<ConfigInst>          &fifo_config_in,
  hls::stream<WeightLoadData0Type> &fifo_depth_conv_weight,
  hls::stream<WeightLoadData1Type> &fifo_conv_weight,
  hls::stream<WeightLoadData2Type> &fifo_bias,
  hls::stream<ConfigInst>          &fifo_config_out
){
#pragma HLS INLINE off 
  // on-chip buffers
  bus_t1 weight_burst_buf1[IN_NUM_T * K_T * K_T / BUS_PACK_FACTOR1];
  bus_t1 weight_burst_buf2[OUT_NUM_T * IN_NUM_T * K_T * K_T / BUS_PACK_FACTOR1];
  bus_t2 bias_burst_buf[OUT_NUM_T / BUS_PACK_FACTOR2];

  // tiling iterators
	uint in_num_iter = 0;
	uint out_num_iter = 0;
	uint in_h_iter = 0;
	uint in_w_iter = 0;
  uint layer_iter = 0;

  // Read instructions
  ConfigInst inst0 = fifo_config_in.read();
  fifo_config_out.write(inst0);
  ConfigInst inst1 = fifo_config_in.read();
  fifo_config_out.write(inst1);
  ConfigInst inst2 = fifo_config_in.read();
  fifo_config_out.write(inst2);
  ConfigInst inst3 = fifo_config_in.read();
  fifo_config_out.write(inst3);
  ConfigInst inst4 = fifo_config_in.read();
  fifo_config_out.write(inst4);

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  bool layer_start = 0;
	bool done = 0;
	// We assum that cin has been pre-padded with zeros
	while(!done){
    if (layer_start){
      inst0 = fifo_config_in.read();
      fifo_config_out.write(inst0);
      inst1 = fifo_config_in.read();
      fifo_config_out.write(inst1);
      inst2 = fifo_config_in.read();
      fifo_config_out.write(inst2);
      inst3 = fifo_config_in.read();
      fifo_config_out.write(inst3);
      inst4 = fifo_config_in.read();
      fifo_config_out.write(inst4);
      layer_start = 0;
    }

    // inst0
    ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
    // inst1
    ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
    // inst2
    ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
    ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
    ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
    ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
    ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
    ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
    ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
    // inst3
    ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
    ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
    ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
    ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);
  
    ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
    ap_uint<1>  CONV_EN          = LAYER_EN[1];
    ap_uint<1>  RELU_EN          = LAYER_EN[2];
    ap_uint<1>  POOL_EN          = LAYER_EN[3];
    ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
  
    // offsets
  	uint weight_offset1 = 0;
    uint weight_offset2 = 0;
  	uint bias_offset = 0;
    weight_offset1 = WEIGHT_OFFSET;
    bias_offset = BIAS_OFFSET;
    if (DEPTH_CONV_EN == 1)
      weight_offset2 = WEIGHT_OFFSET + LAYER_IN_NUM_HW * FILTER_S1 * FILTER_S1;
    else
      weight_offset2 = WEIGHT_OFFSET;

    // bias
    if (CONV_EN == 1){
      // Only write out in the last iteration
      if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
        uint global_bias_offset = bias_offset + out_num_iter;
        memcpy((void*)bias_burst_buf, (void*)&global_bias[global_bias_offset / BUS_PACK_FACTOR2], sizeof(data_t2) * LAYER_OUT_NUM_T);
      }
    }
    // weights1
    if (DEPTH_CONV_EN == 1){
      // load from DRAM
      uint global_weight_offset = weight_offset1 + in_num_iter * FILTER_S1 * FILTER_S1;
      if (FILTER_S1 == 1){
        memcpy((void*)&weight_burst_buf1, (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_IN_NUM_T * 1 * 1);
      } else if (FILTER_S1 == 3){
        memcpy((void*)&weight_burst_buf1, (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_IN_NUM_T * 3 * 3);
      }
    }
    // weights2
    if (CONV_EN == 1){
      uint global_weight_offset = weight_offset2 + out_num_iter * LAYER_IN_NUM_HW * FILTER_S2 * FILTER_S2 + in_num_iter * LAYER_OUT_NUM_T * FILTER_S2 * FILTER_S2;
      if (FILTER_S2 == 1){
        memcpy((void*)&weight_burst_buf2[0], (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_OUT_NUM_T * LAYER_IN_NUM_T * 1 * 1);
      } else if (FILTER_S2 == 3){
        memcpy((void*)&weight_burst_buf2[0], (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_OUT_NUM_T * LAYER_IN_NUM_T * 3 * 3);
      }
    } 

    weight_load_depth_conv_weight_write(weight_burst_buf1, fifo_depth_conv_weight, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
    weight_load_conv_weight_write(weight_burst_buf2, fifo_conv_weight, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
    weight_load_bias_write(bias_burst_buf, fifo_bias, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);

    in_num_iter += LAYER_IN_NUM_T;
    if (in_num_iter >= LAYER_IN_NUM){
      in_num_iter = 0;
      in_h_iter += LAYER_IN_H_T;
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          out_num_iter += LAYER_OUT_NUM_T;
          if (out_num_iter >= LAYER_OUT_NUM){
            out_num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    }
  }
}

/**
 * Function name: inter_load
 * Function description: This function loads intermediate results buffered on-chip from the previous batch.
 */
void inter_load(
  hls::stream<CinLoadData0Type>     &fifo_cin,
  hls::stream<InterWriteData1Type>  &fifo_inter,
  hls::stream<ConfigInst>           &fifo_config_in,
  hls::stream<InterLoadData0Type>   &fifo_cout,
  hls::stream<ConfigInst>           &fifo_config_out
){
#pragma HLS INLINE off
  // tiling iterators
  uint in_num_iter = 0;
  uint out_num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;

  // Read instructions
  ConfigInst inst0 = fifo_config_in.read();
  fifo_config_out.write(inst0);
  ConfigInst inst1 = fifo_config_in.read();
  fifo_config_out.write(inst1);
  ConfigInst inst2 = fifo_config_in.read();
  fifo_config_out.write(inst2);
  ConfigInst inst3 = fifo_config_in.read();
  fifo_config_out.write(inst3);
  ConfigInst inst4 = fifo_config_in.read();
  fifo_config_out.write(inst4);

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  bool layer_start = 0;
  bool done = 0;
  while(!done){
    if (layer_start){
      inst0 = fifo_config_in.read();
      fifo_config_out.write(inst0);
      inst1 = fifo_config_in.read();
      fifo_config_out.write(inst1);
      inst2 = fifo_config_in.read();
      fifo_config_out.write(inst2);
      inst3 = fifo_config_in.read();
      fifo_config_out.write(inst3);
      inst4 = fifo_config_in.read();
      fifo_config_out.write(inst4);
      layer_start = 0;
    }

    // inst0
    ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
    // inst1
    ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
    // inst2
    ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
    ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
    ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
    ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
    ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
    ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16); 
    ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
    // inst3
    ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
    ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
    ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
    ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);
  
    ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
    ap_uint<1>  CONV_EN          = LAYER_EN[1];
    ap_uint<1>  RELU_EN          = LAYER_EN[2];
    ap_uint<1>  POOL_EN          = LAYER_EN[3];
    ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
    ap_uint<1>  BIAS_EN          = LAYER_EN[5];
    ap_uint<1>  INTER_LOAD_EN    = LAYER_EN[6];
    ap_uint<1>  INTER_WRITE_EN   = LAYER_EN[7];
  
    uint FILTER_S = (DEPTH_CONV_EN == 1)? (uint)FILTER_S1: (CONV_EN == 1)? (uint)FILTER_S2: 1;
    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);

#ifdef DEBUG  
    uint fifo_read_cnt = 0;
#endif  

    if ((max_pool && out_num_iter == 0) || conv2d || separable_conv){
      switch(INTER_LOAD_EN){
        case 0:
        {
          int o = 0;
          int h = 0;
          int w = 0;
          bool done1 = 0;
          while(!done1){
#pragma HLS PIPELINE II=1
            CinLoadData0Type tmp = fifo_cin.read();
            fifo_cout.write(tmp);
            w++;
            if (w == LAYER_IN_W_T + FILTER_S - 1){
              w = 0;
              h++;
              if (h == LAYER_IN_H_T + FILTER_S - 1){
                h = 0;
                o++;
                if (o == LAYER_IN_NUM_T / DEPTH_CONV_LANE){
                  o = 0;
                  done1 = 1;
                }
              }
            }
          }
          break;
        }
        case 1:
        {
          int o = 0;
          int h = 0;
          int w = 0;
          bool done1 = 0;
          while(!done1){
#pragma HLS PIPELINE II=1
            InterLoadData0Type tmp;
            if ((h >= int(FILTER_S / 2)) && (h < LAYER_IN_H_T + int(FILTER_S / 2)) 
              && (w >= int(FILTER_S / 2)) && (w < LAYER_IN_W_T + int(FILTER_S / 2))){
              tmp = fifo_inter.read();
            } else {
              tmp = 0;
            }
            fifo_cout.write(tmp);              

            w++;
            if (w == LAYER_IN_W_T + FILTER_S - 1){
              w = 0;
              h++;
              if (h == LAYER_IN_H_T + FILTER_S - 1){
                h = 0;
                o++;
                if (o == LAYER_IN_NUM_T / INTER_LOAD_LANE){
                  o = 0;
                  done1 = 1;
                }
              }
            }
          }
          break;
        }
      }
    }

    in_num_iter += LAYER_IN_NUM_T;
    if (in_num_iter >= LAYER_IN_NUM){
      in_num_iter = 0;
      in_h_iter += LAYER_IN_H_T;
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          out_num_iter += LAYER_OUT_NUM_T;
          if (out_num_iter >= LAYER_OUT_NUM){
            out_num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    }
  }
}

/**
* Function name: depth_conv
* Function description: This function performs depthwise convolution.
*/
void depth_conv(
  hls::stream<CinLoadData0Type>     &fifo_cin,
  hls::stream<WeightLoadData0Type>  &fifo_weight,
  hls::stream<ConfigInst>           &fifo_config_in,
  hls::stream<DepthConvData0Type>   &fifo_cout,
  hls::stream<ConfigInst>           &fifo_config_out
){
#pragma HLS INLINE off
  // tiling iterators
  uint in_num_iter = 0;
  uint out_num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;

  // Read instructions
  ConfigInst inst0 = fifo_config_in.read();
  fifo_config_out.write(inst0);
  ConfigInst inst1 = fifo_config_in.read();
  fifo_config_out.write(inst1);
  ConfigInst inst2 = fifo_config_in.read();
  fifo_config_out.write(inst2);
  ConfigInst inst3 = fifo_config_in.read();
  fifo_config_out.write(inst3);
  ConfigInst inst4 = fifo_config_in.read();
  fifo_config_out.write(inst4);

  // dump write instructions
  fifo_config_out.write(inst0);
  fifo_config_out.write(inst1);
  fifo_config_out.write(inst2);
  fifo_config_out.write(inst3);
  fifo_config_out.write(inst4);

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  bool layer_start = 0;
  bool done = 0;
  while(!done){
    if (layer_start){
      inst0 = fifo_config_in.read();
      fifo_config_out.write(inst0);
      inst1 = fifo_config_in.read();
      fifo_config_out.write(inst1);
      inst2 = fifo_config_in.read();
      fifo_config_out.write(inst2);
      inst3 = fifo_config_in.read();
      fifo_config_out.write(inst3);
      inst4 = fifo_config_in.read();
      fifo_config_out.write(inst4);
     
      layer_start = 0;
    }

    // inst0
    ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
    // inst1
    ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
    // inst2
    ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
    ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
    ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
    ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
    ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
    ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16); 
    ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
    // inst3
    ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
    ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
    ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
    ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);
  
    ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
    ap_uint<1>  CONV_EN          = LAYER_EN[1];
    ap_uint<1>  RELU_EN          = LAYER_EN[2];
    ap_uint<1>  POOL_EN          = LAYER_EN[3];
    ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
  
    data_t1 weight_buf[IN_NUM_T / DEPTH_CONV_LANE][DEPTH_CONV_LANE][K_T][K_T];
#pragma HLS ARRAY_PARTITION variable=weight_buf dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight_buf dim=3 complete
#pragma HLS ARRAY_PARTITION variable=weight_buf dim=4 complete
  
    uint FILTER_S = (DEPTH_CONV_EN == 1)? (uint)FILTER_S1: (CONV_EN == 1)? (uint)FILTER_S2: 1;
    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
  
    switch(DEPTH_CONV_EN){
      case 0:
        // bypass this module
        if ((max_pool && out_num_iter == 0) || conv2d){
          int o = 0;
          int h = 0;
          int w = 0;
          bool done1 = 0;
          while(!done1){
#pragma HLS PIPELINE II=1
            CinLoadData0Type tmp = fifo_cin.read();
            fifo_cout.write(tmp);
            w++;
            if (w == LAYER_IN_W_T + FILTER_S - 1){
              w = 0;
              h++;
              if (h == LAYER_IN_H_T + FILTER_S - 1){
                h = 0;
                o++;
                if (o == LAYER_IN_NUM_T / DEPTH_CONV_LANE){
                  o = 0;
                  done1 = 1;
                }
              }
            }
          }
        }
        break;
      case 1:
      {        
        // load weights
        int o = 0;
        int p = 0;
        int q = 0;
        bool done1 = 0;
        while(!done1){
#pragma HLS PIPELINE II=1
          WeightLoadData0Type w_tmp = fifo_weight.read();
          for (int lane = 0; lane < DEPTH_CONV_LANE; lane++){
#pragma HLS UNROLL
            ap_uint<DATA_W1> utmp = w_tmp(DATA_W1 - 1, 0);
            weight_buf[o][lane][p][q] = Reinterpret<data_t1>(utmp);
            w_tmp = w_tmp >> DATA_W1;                
          }
          
          q++;
          if (q == FILTER_S1){
            q = 0;
            p++;
            if (p == FILTER_S1){
              p = 0;
              o++;
              if (o == LAYER_IN_NUM_T / DEPTH_CONV_LANE){
                o = 0;
                done1 = 1;
              }
            }
          }
        }

        // compute
        if (FILTER_S1 == 1){
          stencil_w1<data_t0, data_t1, IN_NUM_T, IN_H_T, IN_W_T, DEPTH_CONV_LANE, 1, DATA_W0, DATA_W1>(fifo_cin, weight_buf, fifo_cout, (uint)STRIDE, LAYER_IN_NUM_T, LAYER_IN_H_T);
        } else if (FILTER_S1 == 3){
          stencil_w3<data_t0, data_t1, IN_NUM_T, IN_H_T + 2, IN_W_T + 2, DEPTH_CONV_LANE, 3, DATA_W0, DATA_W1>(fifo_cin, weight_buf, fifo_cout, (uint)STRIDE, LAYER_IN_NUM_T, LAYER_IN_H_T + 2);
        }
        break;
      }
    } 

    in_num_iter += LAYER_IN_NUM_T;
    if (in_num_iter >= LAYER_IN_NUM){
      in_num_iter = 0;
      in_h_iter += LAYER_IN_H_T;
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          out_num_iter += LAYER_OUT_NUM_T;
          if (out_num_iter >= LAYER_OUT_NUM){
            out_num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    }

  }
}

/**
* Function name: conv_core
* Function description: This function performs convolution.
*/
void conv_core(
  data_t0 cin[IN_H_T + K_T - 1][IN_W_T + K_T - 1][IN_NUM_T],
  data_t1 weight[OUT_NUM_T][K_T][K_T][IN_NUM_T],
  data_t0 cout[OUT_H_T][OUT_W_T][OUT_NUM_T],
  bool init,
  uint LAYER_IN_NUM_T,
  uint LAYER_OUT_NUM_T,
  uint LAYER_IN_H_T,
  uint LAYER_IN_W_T,
  uint FILTER_S,
  uint STRIDE
){
#pragma HLS INLINE off
  for (int o = 0; o < LAYER_OUT_NUM_T; o++)
    for (int h = 0; h < LAYER_IN_H_T / STRIDE; h++)
	    for (int w = 0; w < LAYER_IN_W_T / STRIDE; w++){
	      if (init){
	        cout[h][w][o] = 0;
	      }
	      for (int i = 0; i < LAYER_IN_NUM_T; i++){
#pragma HLS PIPELINE II=1            
	        for (int p = 0; p < FILTER_S; p++)
	          conv_kernel_loop: for (int q = 0; q < FILTER_S; q++){
              cout[h][w][o] += cin[h * STRIDE + (STRIDE - 1) + p][w * STRIDE + (STRIDE - 1) + q][i] * weight[o][p][q][i];
	          }
        }
	    }
}

/**
* Function name: kernel
* Function description: 
*/
//void kernel(
//  hls::stream<DepthConvData0Type>  &fifo_cin,
//  hls::stream<WeightLoadData1Type> &fifo_weight,
//  hls::stream<ConvData0Type>       &fifo_cout,
//  hls::stream<ConfigInst>          &fifo_config_in,
//  hls::stream<ConfigInst>          &fifo_config_out
//){
//  data_t0 cin_local[IN_H_T + K_T - 1][IN_W_T + K_T - 1][IN_NUM_T];
//#pragma HLS ARRAY_PARTITIOn variable=cin_local dim=3 cyclic factor=8  
//  data_t1 weight_local[OUT_NUM_T][K_T][K_T][IN_NUM_T];
//#pragma HLS ARRAY_PARTITION variable=weight_local dim=4 cyclic factor=8
//  data_t0 cout_local[OUT_H_T][OUT_W_T][OUT_NUM_T];
//#pragma HLS ARRAY_PARTITION variable=cout_local dim=3 cyclic factor=8
//
//  ap_uint<DATA_W0> cout_tmp[RELU_LANE];
//#pragma HLS ARRAY_PARTITION variable=cout_tmp complete 
//
//  uint in_num_iter = 0;
//  uint out_num_iter = 0;
//  uint in_h_iter = 0;
//  uint in_w_iter = 0;
//  uint layer_iter = 0;
//  
//  // Read instructions
//  ConfigInst inst0 = fifo_config_in.read();
//  fifo_config_out.write(inst0);
//  ConfigInst inst1 = fifo_config_in.read();
//  fifo_config_out.write(inst1);
//  ConfigInst inst2 = fifo_config_in.read();
//  fifo_config_out.write(inst2);
//  ConfigInst inst3 = fifo_config_in.read();
//  fifo_config_out.write(inst3);
//  ConfigInst inst4 = fifo_config_in.read();
//  fifo_config_out.write(inst4);
//
//  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);
//
//  bool done = 0;
//  bool layer_start = 0;  
//  while(!done){
//    if (layer_start){
//      // Read instructions
//      inst0 = fifo_config_in.read();
//      fifo_config_out.write(inst0);
//      inst1 = fifo_config_in.read();
//      fifo_config_out.write(inst1);
//      inst2 = fifo_config_in.read();
//      fifo_config_out.write(inst2);
//      inst3 = fifo_config_in.read();
//      fifo_config_out.write(inst3);
//      inst4 = fifo_config_in.read();
//      fifo_config_out.write(inst4);
//
//      layer_start = 0;
//    }
//
//    // inst0
//    ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
//    ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
//    ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
//    ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
//    ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
//    ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
//    // inst1
//    ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
//    ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
//    ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
//    ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
//    ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
//    ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
//    // inst2
//    ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
//    ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
//    ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
//    ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
//    ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
//    ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
//    ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
//    // inst3
//    ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
//    ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
//    ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
//    ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
//    ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);
//
//    ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
//    ap_uint<1>  CONV_EN          = LAYER_EN[1];
//    ap_uint<1>  RELU_EN          = LAYER_EN[2];
//    ap_uint<1>  POOL_EN          = LAYER_EN[3];
//    ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
//  
//    uint FILTER_S = (CONV_EN == 1)? (uint)FILTER_S2: 1;
//    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
//    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
//    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
//    uint stride1 = (DEPTH_CONV_EN == 0)? 1 : (uint)STRIDE;
//    uint stride2 = (DEPTH_CONV_EN == 0)? (uint)STRIDE : 1;
//
//    switch(CONV_EN){
//      case 0:
//        // bypass
//        if (max_pool && out_num_iter == 0){
//          for (int o = 0; o < LAYER_IN_NUM_T / CONV_LANE; o++)
//            for (int h = 0; h < LAYER_IN_H_T + FILTER_S - 1; h++)
//              for (int w = 0; w < LAYER_IN_W_T + FILTER_S - 1; w++){
//#pragma HLS PIPELINE II=1
//                DepthConvData0Type tmp = fifo_cin.read();
//                fifo_cout.write(tmp);              
//              }
//        }
//        
//        break;
//      case 1:
//        for (int i = 0; i < LAYER_IN_NUM_T / CONV_LANE; i++)
//          for (int h = 0; h < LAYER_IN_H_T / stride1 + FILTER_S - 1; h++)
//            for (int w = 0; w < LAYER_IN_W_T / stride1 + FILTER_S - 1; w++){
//#pragma HLS PIPELINE II=1
//              DepthConvData0Type tmp = fifo_cin.read();
//              for (int lane = 0; lane < CONV_LANE; lane++){
//#pragma HLS UNROLL
//                ap_uint<DATA_W0> u32_tmp = tmp(DATA_W0 - 1, 0);
//                cin_local[h][w][i * CONV_LANE + lane] = Reinterpret<data_t0>(u32_tmp);
//                tmp = tmp >> DATA_W0;
//              }
//            }
//
//#ifdef DEBUG
//        if (in_num_iter == 0 && out_num_iter == 0 && in_h_iter == 0 && in_w_iter == 0){
//          ofstream cin_debug("hw_cin_patch.dat");
//          for (int h = 0; h < LAYER_IN_H_T + FILTER_S - 1; h++)
//            for (int w = 0; w < LAYER_IN_W_T + FILTER_S - 1; w++){
//              cin_debug << cin_local[h][w][0] << endl;
//            }
//          cin_debug.close();
//        }
//#endif    
//        for (int o = 0; o < LAYER_OUT_NUM_T; o++)
//          for (int p = 0; p < FILTER_S; p++)
//            for (int q = 0; q < FILTER_S; q++)
//              for (int i = 0; i < LAYER_IN_NUM_T / CONV_LANE; i++){
//#pragma HLS PIPELINE II=1
//                WeightLoadData1Type tmp = fifo_weight.read();
//#ifdef DEBUG
////            if (out_num_iter == 0 && in_h_iter == 0 && in_w_iter == 0){
////              if (o == 0 && p == 0 && q == 0){
////                cout << "conv_core_first_weight: " << Reinterpret<data_t1>(tmp) << "(" << o 
////                  << " " << i << " " << out_num_iter << " " << in_h_iter << " " << in_w_iter << endl;
////              }
////            }
//#endif
//                for (int lane = 0; lane < CONV_LANE; lane++){
//#pragma HLS UNROLL
//                  ap_uint<DATA_W1> u32_tmp = tmp(DATA_W1 - 1, 0);
//                  weight_local[o][p][q][i * CONV_LANE + lane] = Reinterpret<data_t1>(u32_tmp);
//#ifdef DEBUG
////              if (in_num_iter == 0 && out_num_iter == 0 && in_h_iter == 0 && in_w_iter == 0){
////                if (o == 0 && p == 0 && q == 0 && i == 0){
////                  ap_uint<32> debug_tmp = tmp(DATA_W1 - 1, 0);
////                  data_t1 debug_f_tmp = Reinterpret<data_t1>(debug_tmp);
////                  cout << debug_f_tmp;
////                  debug_f_tmp = Reinterpret<data_t1>(tmp(DATA_W1 - 1, 0));
////                  cout << debug_f_tmp;
////                  cout << "conv_core_first_weight: " << weight_local[0][p][q][0] << endl;
////                }
////              }
//#endif              
//                  tmp = tmp >> DATA_W1;
//                }
//              }
//
//#ifdef DEBUG
////    // print out the first filter
////    if (in_num_iter == 0 && out_num_iter == 0 && in_h_iter == 0 && in_w_iter == 0){
////      for (int p = 0; p < FILTER_S; p++)
////        for (int q = 0; q < FILTER_S; q++){
////          cout << "conv_core: " << weight_local[0][p][q][0] << endl;
////        }
////    }
//#endif    
//    
//        conv_core(cin_local, weight_local, cout_local, (in_num_iter == 0), LAYER_IN_NUM_T, LAYER_OUT_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, FILTER_S, stride2);
//
////#ifdef DEBUG
////        if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM && out_num_iter == 0 && in_h_iter == 0 && in_w_iter == 0){
////          ofstream debug_conv("kernel_conv_patch.dat");
////          for (int h = 0; h < OUT_H_T; h++)
////            for (int w = 0; w < OUT_W_T; w++){
////              debug_conv << cout_local[h][w][0] << endl;
////            }
////          debug_conv.close();
////        }
////#endif    
//
//        if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
//          for (int o = 0; o < LAYER_OUT_NUM_T / RELU_LANE; o++)
//            for (int h = 0; h < LAYER_IN_H_T / STRIDE; h++)
//              for (int w = 0; w < LAYER_IN_W_T / STRIDE; w++){
//#pragma HLS PIPELINE II=1
//                for (int lane = 0; lane < CONV_LANE; lane++){
//#pragma HLS UNROLL  
//                  data_t0 tmp_f = cout_local[h][w][o * RELU_LANE + lane];
//                  cout_tmp[lane] = Reinterpret<ap_uint<DATA_W0> >(tmp_f);
//                }
//                ConvData0Type tmp = (
//#if RELU_LANE == 16                
//                  cout_tmp[15], cout_tmp[14], cout_tmp[13], cout_tmp[12],
//                  cout_tmp[11], cout_tmp[10], cout_tmp[9], cout_tmp[8],
//                  cout_tmp[7], cout_tmp[6], cout_tmp[5], cout_tmp[4],
//                  cout_tmp[3], cout_tmp[2], cout_tmp[1], cout_tmp[0]
//#elif RELU_LANE == 8
//                  cout_tmp[7], cout_tmp[6], cout_tmp[5], cout_tmp[4],
//                  cout_tmp[3], cout_tmp[2], cout_tmp[1], cout_tmp[0]
//#elif RELU_LANE == 4
//                  cout_tmp[3], cout_tmp[2], cout_tmp[1], cout_tmp[0]
//#elif RELU_LANE == 2
//                  cout_tmp[1], cout_tmp[0]
//#elif RELU_LANE == 1
//                  cout_tmp[0]
//#endif                
//                );
//                fifo_cout.write(tmp);
//#ifdef DEBUG
////            if (out_num_iter == 0 && in_h_iter == 0 && in_w_iter == 0){
////              if (o == 0 && h == 1 && w == 1){
////                data_t0 f_tmp = Reinterpret<data_t0>(tmp);
////                cout << "conv first one: " << f_tmp << endl;
////              }
////            }
//#endif            
//              }
//        }
//        break;
//    }
//    
//    in_num_iter += LAYER_IN_NUM_T;
//    if (in_num_iter >= LAYER_IN_NUM){
//      in_num_iter = 0;
//      in_h_iter += LAYER_IN_H_T;
//      if (in_h_iter >= LAYER_IN_H){
//        in_h_iter = 0;
//        in_w_iter += LAYER_IN_W_T;
//        if (in_w_iter >= LAYER_IN_W){
//          in_w_iter = 0;
//          out_num_iter += LAYER_OUT_NUM_T;
//          if (out_num_iter >= LAYER_OUT_NUM){
//            out_num_iter = 0;
//            layer_iter += 1;
//            layer_start = 1;
//            if (layer_iter == LAYER_BATCH){
//              layer_iter = 0;
//              done = 1;
//            }
//          }
//        }
//      }
//    }
//  }
//}

/**
* Function name: conv
* Function description: 
*/
void conv(
  hls::stream<DepthConvData0Type>  &fifo_cin,
  hls::stream<WeightLoadData1Type> &fifo_weight,
  hls::stream<ConfigInst>          &fifo_config_in,
  hls::stream<ConvData0Type>       &fifo_cout,
  hls::stream<ConfigInst>          &fifo_config_out
){
#pragma HLS INLINE off 
  uint in_num_iter = 0;
  uint out_num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;
  
  // Dummpy first read
  ConfigInst inst0 = fifo_config_in.read();
  ConfigInst inst1 = fifo_config_in.read();
  ConfigInst inst2 = fifo_config_in.read();
  ConfigInst inst3 = fifo_config_in.read();
  ConfigInst inst4 = fifo_config_in.read();

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  // inst0
  ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
  // inst1
  ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
  ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
  ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
  ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
  ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
  // inst2
  ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
  ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
  ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
  ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
  ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
  ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
  ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
  // inst3
  ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
  ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
  ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
  ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
  ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

  ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
  ap_uint<1>  CONV_EN          = LAYER_EN[1];
  ap_uint<1>  RELU_EN          = LAYER_EN[2];
  ap_uint<1>  POOL_EN          = LAYER_EN[3];
  ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
  
  uint FILTER_S = (CONV_EN == 1)? (uint)FILTER_S2: 1;
  bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
  bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
  bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);

  switch(CONV_EN){
    case 0:
      // bypass
      for (int layer_iter = 0; layer_iter < LAYER_BATCH; layer_iter++){
        // Read instructions
        inst0 = fifo_config_in.read();
        fifo_config_out.write(inst0);
        inst1 = fifo_config_in.read();
        fifo_config_out.write(inst1);
        inst2 = fifo_config_in.read();
        fifo_config_out.write(inst2);
        inst3 = fifo_config_in.read();
        fifo_config_out.write(inst3);
        inst4 = fifo_config_in.read();
        fifo_config_out.write(inst4);

        // inst0
        LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
        LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
        LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
        LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
        LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
        LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
        // inst1
        LAYER_IN_NUM     = inst1(32*0+31, 32*0);
        LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
        LAYER_IN_H       = inst1(32*2+31, 32*2);
        LAYER_IN_W       = inst1(32*3+31, 32*3);
        LAYER_OUT_H      = inst1(32*4+31, 32*4);
        LAYER_OUT_W      = inst1(32*5+31, 32*5);
        // inst2
        CIN_OFFSET       = inst2(32*0+31, 32*0);
        WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
        BIAS_OFFSET      = inst2(32*2+31, 32*2);
        COUT_OFFSET      = inst2(32*3+31, 32*3);
        FILTER_S1        = inst2(32*4+15, 32*4);
        FILTER_S2        = inst2(32*4+31, 32*4+16);
        STRIDE           = inst2(32*5+31, 32*5);
        // inst3
        LAYER_EN         = inst3(32*0+31, 32*0);
        LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
        LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
        LAYER_IN_H_T     = inst3(32*3+31, 32*3);
        LAYER_IN_W_T     = inst3(32*4+31, 32*4);
    
        DEPTH_CONV_EN    = LAYER_EN[0];
        CONV_EN          = LAYER_EN[1];
        RELU_EN          = LAYER_EN[2];
        POOL_EN          = LAYER_EN[3];
        UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
      
        FILTER_S = (CONV_EN == 1)? (uint)FILTER_S2: 1;
        separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
        conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
        max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);

        int in_h_iter = 0;
        int in_w_iter = 0;
        int out_num_iter = 0;
        int in_num_iter = 0;
        bool done1 = 0;
        while(!done1){
          if (max_pool && out_num_iter == 0){
            int o = 0;
            int h = 0;
            int w = 0;
            bool done2 = 0;
            while(!done2){
#pragma HLS PIPELINE II=1
              DepthConvData0Type tmp = fifo_cin.read();
              fifo_cout.write(tmp);              

              w++;
              if (w == LAYER_IN_W_T + FILTER_S - 1){
                w = 0;
                h++;
                if (h == LAYER_IN_H_T + FILTER_S - 1){
                  h = 0;
                  o++;
                  if (o == LAYER_IN_NUM_T / CONV_LANE){
                    o = 0;
                    done2 = 1;
                  }
                }
              }
            }
          }
          in_num_iter += LAYER_IN_NUM_T;
          if (in_num_iter == LAYER_IN_NUM){
            in_num_iter = 0;
            out_num_iter += LAYER_OUT_NUM_T;
            if (out_num_iter == LAYER_OUT_NUM){
              out_num_iter = 0;
              in_w_iter += LAYER_IN_W_T;
              if (in_w_iter == LAYER_IN_W){
                in_w_iter = 0;
                in_h_iter += LAYER_IN_H_T;
                if (in_h_iter == LAYER_IN_H){
                  in_h_iter = 0;
                  done1 = 1;
                }
              }
            }
          }

        }
      }
      break;
    case 1:
      kernel(fifo_cin, fifo_weight, fifo_cout, fifo_config_in, fifo_config_out);
      break;
  }
}

/**
* Function name: relu
* Function description: This functions performs ReLu operation and adds bias to cout results.
*/
void relu(
  hls::stream<ConvData0Type>        &fifo_cin,
  hls::stream<WeightLoadData2Type>  &fifo_bias,
  hls::stream<ConfigInst>           &fifo_config_in,
  hls::stream<ReluData0Type>        &fifo_cout,
  hls::stream<ConfigInst>           &fifo_config_out
){
#pragma INLINE off
  // tiling iterators
  uint in_num_iter = 0;
  uint out_num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;

  // Read instructions
  ConfigInst inst0 = fifo_config_in.read();
  fifo_config_out.write(inst0);
  ConfigInst inst1 = fifo_config_in.read();
  fifo_config_out.write(inst1);
  ConfigInst inst2 = fifo_config_in.read();
  fifo_config_out.write(inst2);
  ConfigInst inst3 = fifo_config_in.read();
  fifo_config_out.write(inst3);
  ConfigInst inst4 = fifo_config_in.read();
  fifo_config_out.write(inst4);

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  bool layer_start = 0;
  bool done = 0;
  while(!done){

    if (layer_start){
      inst0 = fifo_config_in.read();
      fifo_config_out.write(inst0);
      inst1 = fifo_config_in.read();
      fifo_config_out.write(inst1);
      inst2 = fifo_config_in.read();
      fifo_config_out.write(inst2);
      inst3 = fifo_config_in.read();
      fifo_config_out.write(inst3);
      inst4 = fifo_config_in.read();
      fifo_config_out.write(inst4);

      layer_start = 0;
    }

    // inst0
    ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
    // inst1
    ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
    // inst2
    ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
    ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
    ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
    ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
    ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
    ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
    ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
    // inst3
    ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
    ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
    ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
    ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);
  
    ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
    ap_uint<1>  CONV_EN          = LAYER_EN[1];
    ap_uint<1>  RELU_EN          = LAYER_EN[2];
    ap_uint<1>  POOL_EN          = LAYER_EN[3];
    ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
    ap_uint<1>  BIAS_EN          = LAYER_EN[5];
 
    data_t2 bias_buf[OUT_NUM_T / RELU_LANE][RELU_LANE]; 
#pragma HLS ARRAY_PARTITION variable=bias_buf dim=2 complete   
    data_t0 cin_buf[RELU_LANE];  
    ap_uint<DATA_W0> cout_buf[RELU_LANE];
#pragma HLS ARRAY_PARTITION variable=cin_buf complete
#pragma HLS ARRAY_PARTITION variable=cout_buf complete
  
    uint FILTER_S = 1;
    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
    uint stride = (max_pool == 1)? 1 : (uint)STRIDE; 
    bool en = RELU_EN || BIAS_EN;

#ifdef DEBUG
    uint relu_cout_cnt = 0;
    ofstream relu_data;
    relu_data.open("relu_patch.dat", ios::app);
#endif
    switch(en){
      case 0: 
        // bypass this module
        if ((max_pool && out_num_iter == 0) || (!max_pool && (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM))){
          int o = 0;
          int h = 0;
          int w = 0;
          bool done1 = 0;

          int w_bound = LAYER_IN_W_T / stride + FILTER_S - 1;
          int h_bound = LAYER_IN_H_T / stride + FILTER_S - 1;
          while(!done1){
#pragma HLS PIPELINE II=1
            ConvData0Type tmp = fifo_cin.read();
            fifo_cout.write(tmp);                            

            w++;
            if (w == w_bound){
              w = 0;
              h++;
              if (h == h_bound){
                h = 0;
                o++;
                if (o == LAYER_OUT_NUM_T / RELU_LANE){
                  o = 0;
                  done1 = 1;
                }
              }
            }
          }
        }
        break;
      case 1:
      {
        if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
          for (int o = 0; o < LAYER_OUT_NUM_T / RELU_LANE; o++){
#pragma HLS PIPELINE II=1
            WeightLoadData2Type tmp = fifo_bias.read(); 
            for (int lane = 0; lane < RELU_LANE; lane++){
#pragma HLS UNROLL
              ap_uint<DATA_W2> u32_tmp = tmp(DATA_W2 - 1, 0);
              bias_buf[o][lane] = Reinterpret<data_t2>(u32_tmp);
              tmp = tmp >> DATA_W2;
            }
          }

          int o = 0;
          int h = 0;
          int w = 0;
          bool done2 = 0;

          int w_bound = LAYER_IN_W_T / STRIDE;
          int h_bound = LAYER_IN_H_T / STRIDE;

          while(!done2){
#pragma HLS PIPELINE II=1
            ConvData0Type cin_tmp = fifo_cin.read();
            for (int lane = 0; lane < RELU_LANE; lane++){
#pragma HLS UNROLL
              ap_uint<DATA_W0> u32_tmp = cin_tmp(DATA_W0 - 1, 0);
              cin_buf[lane] = Reinterpret<data_t0>(u32_tmp);
              cin_tmp = cin_tmp >> DATA_W0;
            }
            for (int lane = 0; lane < RELU_LANE; lane++){
#pragma HLS UNROLL              
              data_t0 tmp = cin_buf[lane] + bias_buf[o][lane];
              if (RELU_EN)
                tmp = max(0, tmp);
              cout_buf[lane] = Reinterpret<ap_uint<DATA_W0> >(tmp);
            }
            // write out
            ReluData0Type wide_tmp = (
#if RELU_LANE == 16
              cout_buf[15], cout_buf[14], cout_buf[13], cout_buf[12],
              cout_buf[11], cout_buf[10], cout_buf[9], cout_buf[8],
              cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif RELU_LANE == 8
              cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif RELU_LANE == 4
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif RELU_LANE == 2              
              cout_buf[1], cout_buf[0]
#elif RELU_LANE == 1
              cout_buf[0]
#endif                
            );
            fifo_cout.write(wide_tmp);

            w++;
            if (w == w_bound){
              w = 0;
              h++;
              if (h == h_bound){
                h = 0;
                o++;
                if (o == LAYER_OUT_NUM_T / RELU_LANE){
                  o = 0;
                  done2 = 1;
                }
              }
            }
          }
        }
        break;
      }
    }

    in_num_iter += LAYER_IN_NUM_T;
    if (in_num_iter >= LAYER_IN_NUM){
      in_num_iter = 0;
      in_h_iter += LAYER_IN_H_T;
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          out_num_iter += LAYER_OUT_NUM_T;
          if (out_num_iter >= LAYER_OUT_NUM){
            out_num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    } 
  }
}

/**
* Function name: pool
* Function description: This functions performs max-pooling operation.
*/
void pool(
  hls::stream<ReluData0Type>  &fifo_cin,
  hls::stream<ConfigInst>     &fifo_config_in,
  hls::stream<PoolData0Type>  &fifo_cout,
  hls::stream<ConfigInst>     &fifo_config_out
){
  // tiling factors
  uint in_num_iter = 0;
  uint out_num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;

  // Read instructions
  ConfigInst inst0 = fifo_config_in.read();
  fifo_config_out.write(inst0);
  ConfigInst inst1 = fifo_config_in.read();
  fifo_config_out.write(inst1);
  ConfigInst inst2 = fifo_config_in.read();
  fifo_config_out.write(inst2);
  ConfigInst inst3 = fifo_config_in.read();
  fifo_config_out.write(inst3);
  ConfigInst inst4 = fifo_config_in.read();
  fifo_config_out.write(inst4);

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  bool layer_start = 0;
  bool done = 0;
  while(!done){

    if (layer_start){
      inst0 = fifo_config_in.read();
      fifo_config_out.write(inst0);
      inst1 = fifo_config_in.read();
      fifo_config_out.write(inst1);
      inst2 = fifo_config_in.read();
      fifo_config_out.write(inst2);
      inst3 = fifo_config_in.read();
      fifo_config_out.write(inst3);
      inst4 = fifo_config_in.read();
      fifo_config_out.write(inst4);

      layer_start = 0;
    }

    // inst0
    ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
    // inst1
    ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
    // inst2
    ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
    ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
    ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
    ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
    ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
    ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
    ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
    // inst3
    ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
    ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
    ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
    ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);
  
    ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
    ap_uint<1>  CONV_EN          = LAYER_EN[1];
    ap_uint<1>  RELU_EN          = LAYER_EN[2];
    ap_uint<1>  POOL_EN          = LAYER_EN[3];
    ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
  
    bool en = POOL_EN;
    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);

    switch(en){
      case 0:       
        if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
          int o = 0;
          int h = 0;
          int w = 0;
          bool done1 = 0;

          int w_bound = LAYER_IN_W_T / STRIDE;
          int h_bound = LAYER_IN_H_T / STRIDE;

          while(!done1){
#pragma HLS PIPELINE II=1
            PoolData0Type tmp = fifo_cin.read();
            fifo_cout.write(tmp);

            w++;
            if (w == w_bound){
              w = 0;
              h++;
              if (h == h_bound){
                h = 0;
                o++;
                if (o == LAYER_OUT_NUM_T / POOL_LANE){
                  o = 0;
                  done1 = 1;
                }
              }
            }
          }          
        }
        break;
      case 1:
        if ((max_pool && out_num_iter == 0) || (!max_pool && (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM))){
          maxpool_w2 <data_t0, OUT_H_T, OUT_W_T, POOL_LANE, 2, DATA_W0> (fifo_cin, fifo_cout, STRIDE, POOL_EN, LAYER_OUT_NUM_T, LAYER_IN_H_T);
        }
        break;
    } 
    
    in_num_iter += LAYER_IN_NUM_T;
    if (in_num_iter >= LAYER_IN_NUM){
      in_num_iter = 0;
      in_h_iter += LAYER_IN_H_T;
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          out_num_iter += LAYER_OUT_NUM_T;
          if (out_num_iter >= LAYER_OUT_NUM){
            out_num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    }
  }
}

/**
* Function name: inter_write
* Function description: This functions writes out intermediate results to on-chip buffers.
*                       Currently deprecated.
*/
void inter_write(
  hls::stream<PoolData0Type> &fifo_cin,
  hls::stream<ConfigInst>    &fifo_config_in,
  hls::stream<InterWriteData0Type> &fifo_cout,
//  hls::stream<InterWriteData1Type> &fifo_inter,
  hls::stream<ConfigInst>    &fifo_config_out
){
#pragma HLS INLINE off
  // tiling iterators
  uint in_num_iter = 0;
  uint out_num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;

  // Read instructions
  ConfigInst inst0 = fifo_config_in.read();
  fifo_config_out.write(inst0);
  ConfigInst inst1 = fifo_config_in.read();
  fifo_config_out.write(inst1);
  ConfigInst inst2 = fifo_config_in.read();
  fifo_config_out.write(inst2);
  ConfigInst inst3 = fifo_config_in.read();
  fifo_config_out.write(inst3);
  ConfigInst inst4 = fifo_config_in.read();
  fifo_config_out.write(inst4);

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  bool layer_start = 0;
  bool done = 0;
  while(!done){
    if (layer_start){
      inst0 = fifo_config_in.read();
      fifo_config_out.write(inst0);
      inst1 = fifo_config_in.read();
      fifo_config_out.write(inst1);
      inst2 = fifo_config_in.read();
      fifo_config_out.write(inst2);
      inst3 = fifo_config_in.read();
      fifo_config_out.write(inst3);
      inst4 = fifo_config_in.read();
      fifo_config_out.write(inst4);

      layer_start = 0;
    }

    // inst0
    ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
    // inst1
    ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
    ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
    ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
    ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
    ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
    // inst2
    ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
    ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
    ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
    ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
    ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
    ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16); 
    ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
    // inst3
    ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
    ap_uint<32> LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
    ap_uint<32> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
    ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
    ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);
  
    ap_uint<1>  DEPTH_CONV_EN    = LAYER_EN[0];
    ap_uint<1>  CONV_EN          = LAYER_EN[1];
    ap_uint<1>  RELU_EN          = LAYER_EN[2];
    ap_uint<1>  POOL_EN          = LAYER_EN[3];
    ap_uint<1>  UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
    ap_uint<1>  BIAS_EN          = LAYER_EN[5]; 
    ap_uint<1>  INTER_LOAD_EN    = LAYER_EN[6];
    ap_uint<1>  INTER_WRITE_EN   = LAYER_EN[7];
  
    uint FILTER_S = (DEPTH_CONV_EN == 1)? (uint)FILTER_S1: (CONV_EN == 1)? (uint)FILTER_S2: 1;
    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
    bool en = POOL_EN || (POOL_EN == 0 && STRIDE == 2);
  
#ifdef DEBUG  
    int fifo_write_cnt = 0;
#endif  

    switch(en){
      case 0:
        if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
          int o = 0;
          int h = 0;
          int w = 0;
          bool done1 = 0;
          while(!done1){
#pragma HLS PIPELINE II=1
            PoolData0Type tmp = fifo_cin.read();
            if (!(in_h_iter + h < LAYER_OUT_H && in_w_iter + w < LAYER_OUT_W)){
              tmp = 0;
            }
            if (INTER_WRITE_EN == 0){
              fifo_cout.write(tmp);
            }
            else {
//            fifo_inter.write(tmp);
#ifdef DEBUG                  
              fifo_write_cnt++;
#endif                  
            }

            w++;
            if (w == LAYER_IN_W_T){
              w = 0;
              h++;
              if (h == LAYER_IN_H_T){
                h = 0;
                o++;
                if (o == LAYER_OUT_NUM_T / DEPTH_CONV_LANE){
                  o = 0;
                  done1 = 1;
                }
              }
            }
          }
        }
        break;
      case 1:
        if ((max_pool && out_num_iter == 0) || (!max_pool && (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM))){
          int o = 0;
          int h = 0;
          int w = 0;
          bool done1 = 0;
          while(!done1){
#pragma HLS PIPELINE II=1
            PoolData0Type tmp = fifo_cin.read();
            if (!(in_h_iter / 2 + h < LAYER_OUT_H && in_w_iter / 2 + w < LAYER_OUT_W)){
              tmp = 0;
            }
            if (INTER_WRITE_EN == 0)
              fifo_cout.write(tmp);
            else{
//              fifo_inter.write(tmp);              
            }
            w++;
            if (w == LAYER_IN_W_T / 2){
              w = 0;
              h++;
              if (h == LAYER_IN_H_T / 2){
                h = 0;
                o++;
                if (o == LAYER_OUT_NUM_T / INTER_LOAD_LANE){
                  o = 0;
                  done1 = 1;
                }
              }
            }
          }
        }
        break;
    }

    in_num_iter += LAYER_IN_NUM_T;
    if (in_num_iter >= LAYER_IN_NUM){
      in_num_iter = 0;
      in_h_iter += LAYER_IN_H_T;
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          out_num_iter += LAYER_OUT_NUM_T;
          if (out_num_iter >= LAYER_OUT_NUM){
            out_num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    }
  }
}

/**
* Function name: cout_write_fifo_read
* Function description: This function reads cout data.
*/
void cout_write_fifo_read(
  bus_t0 cout_burst_buf[],
  hls::stream<PoolData0Type>  &fifo_cout,  
  bool en,
  bool max_pool,
  uint LAYER_IN_NUM,
  uint LAYER_OUT_H,
  uint LAYER_OUT_W,
  uint LAYER_IN_NUM_T,
  uint LAYER_OUT_NUM_T,
  uint LAYER_IN_H_T,
  uint LAYER_IN_W_T,
//  uint num_iter,
  uint in_h_iter,
  uint in_w_iter
){
  PoolData0Type cout_buf[DATA_SEL_FACTOR0];
#pragma HLS ARRAY_PARTITION variable=cout_buf complete

#ifdef DEBUG
  ofstream cout_debug;
  cout_debug.open("hw_cout_write_patch.dat", ios::app);
#endif        

    switch(en){
      case 0:
        {
          int o = 0;
          int h = 0;
          int w = 0;
          bool done = 0;
          while(!done){
#pragma HLS PIPELINE II=1       
#pragma HLS DEPENDENCE INTER false variable=cout_burst_buf
            uint local_cout_idx = h * LAYER_IN_W_T * LAYER_OUT_NUM_T + w * LAYER_OUT_NUM_T + o * POOL_LANE;
            bus_t0 wide_tmp = cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0];
            for (int lane = 0; lane < DATA_SEL_FACTOR0; lane++){
#pragma HLS UNROLL
              cout_buf[lane] = wide_tmp(DATA_W0 * POOL_LANE - 1, 0);
              wide_tmp = wide_tmp >> DATA_W0 * POOL_LANE;
            }
            PoolData0Type tmp = fifo_cout.read();                  
            if (in_h_iter + h < LAYER_OUT_H && in_w_iter + w < LAYER_OUT_W)
              cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = tmp;
            else
              cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = tmp;
          
            bus_t0 wide_pack = (
#if DATA_SEL_FACTOR0 == 1
              cout_buf[0]
#elif DATA_SEL_FACTOR0 == 2
              cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 4
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 8
              cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 16
              cout_buf[15], cout_buf[14], cout_buf[13], cout_buf[12],
              cout_buf[11], cout_buf[10], cout_buf[9], cout_buf[8],
              cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#endif                  
            );
            cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0] = wide_pack;

            w++;
            if (w == LAYER_IN_W_T){
              w = 0;
              h++;
              if (h == LAYER_IN_H_T){
                h = 0;
                o++;
                if (o == LAYER_OUT_NUM_T / POOL_LANE){
                  o = 0;
                  done = 1;
                }
              }
            }
          } 
//        }
        break;
        }
      case 1:
        {
          int o = 0;
          int h = 0;
          int w = 0;
          bool done = 0;
          while(!done){
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE INTER false variable=cout_burst_buf                  
            uint local_cout_idx = h * (LAYER_IN_W_T / 2) * LAYER_OUT_NUM_T + w * LAYER_OUT_NUM_T + o * POOL_LANE;
            bus_t0 wide_tmp = cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0];
            for (int lane = 0; lane < DATA_SEL_FACTOR0; lane++){
#pragma HLS UNROLL
              cout_buf[lane] = wide_tmp(DATA_W0 * POOL_LANE - 1, 0);
              wide_tmp = wide_tmp >> DATA_W0 * POOL_LANE;
            }
            PoolData0Type tmp = fifo_cout.read();
            if (in_h_iter / 2 + h < LAYER_OUT_H && in_w_iter / 2 + w < LAYER_OUT_W)
              cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = tmp;
            else                    
              cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = 0;
            bus_t0 wide_pack = (
#if DATA_SEL_FACTOR0 == 1
              cout_buf[0]
#elif DATA_SEL_FACTOR0 == 2
              cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 4
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 8
              cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 16
              cout_buf[15], cout_buf[14], cout_buf[13], cout_buf[12],
              cout_buf[11], cout_buf[10], cout_buf[9], cout_buf[8],
              cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
              cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#endif                  
            );
            cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0] = wide_pack;

            w++;
            if (w == LAYER_IN_W_T / 2){
              w = 0;
              h++;
              if (h == LAYER_IN_H_T / 2){
                h = 0;
                o++;
                if (o == LAYER_OUT_NUM_T / POOL_LANE){
                  o = 0;
                  done = 1;
                }
              }
            }
          }        
        }
        break;
    }
}

/**
* Function name: cout_write_ddr_write
* Function description: This function writes out cout results to off-chip DRAM.
*/
void cout_write_ddr_write(
  bus_t0 cout_burst_buf[],
  bus_t0 *global_cout,
  bool en,
  bool max_pool,
  uint num_iter,
  uint in_h_iter,
  uint in_w_iter,
  uint LAYER_IN_NUM,
  uint LAYER_OUT_NUM,
  uint LAYER_IN_NUM_T,
  uint LAYER_OUT_NUM_T,
  uint LAYER_IN_H_T,
  uint LAYER_IN_W_T,
  uint LAYER_OUT_H_HW,
  uint LAYER_OUT_W_HW,
  uint cout_offset,
  bool run
){
  if (run){
    switch(en){
      case 0:
        {
          // write out 
          for (int hh  = 0; hh < LAYER_IN_H_T; hh++){
            uint h = in_h_iter + hh;
            uint global_cout_idx = num_iter / LAYER_OUT_NUM_T * LAYER_OUT_H_HW * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + h * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + in_w_iter * LAYER_OUT_NUM_T + cout_offset;
            uint local_cout_idx = hh * LAYER_IN_W_T * LAYER_OUT_NUM_T;
            memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T * LAYER_OUT_NUM_T);
          }
        }
        break;
      case 1:
        {
          for (int hh = 0; hh < LAYER_IN_H_T / 2; hh++){
            uint h = in_h_iter / 2 + hh;
            uint global_cout_idx;
            global_cout_idx = num_iter / LAYER_OUT_NUM_T * LAYER_OUT_H_HW * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + h * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + in_w_iter / 2 * LAYER_OUT_NUM_T + cout_offset;
            uint local_cout_idx = hh * LAYER_IN_W_T / 2 * LAYER_OUT_NUM_T;
            memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T / 2 * LAYER_OUT_NUM_T);
          }
        }
        break;
    }
  }
}

/**
* Function name: cout_write
* Function description: This function collects and writes out cout results.
*/
void cout_write(
  hls::stream<PoolData0Type>  &fifo_cout,
  hls::stream<ConfigInst>     &fifo_config_in, 
  bus_t0                      *global_cout
){
  bus_t0 cout_burst_buf_ping[OUT_H_T * OUT_W_T * OUT_NUM_T / BUS_PACK_FACTOR0];
  bus_t0 cout_burst_buf_pong[OUT_H_T * OUT_W_T * OUT_NUM_T / BUS_PACK_FACTOR0];

  // iterators
  uint num_iter = 0;
  uint in_h_iter = 0;
  uint in_w_iter = 0;
  uint layer_iter = 0;

  uint cout_offset = 0;

  uint num_iter_prev = 0;
  uint in_h_iter_prev = 0;
  uint in_w_iter_prev = 0;

  // parameters
  // inst0
  ap_uint<32> LAYER_IN_NUM_HW; 
  ap_uint<32> LAYER_OUT_NUM_HW;
  ap_uint<32> LAYER_IN_H_HW;
  ap_uint<32> LAYER_IN_W_HW;  
  ap_uint<32> LAYER_OUT_H_HW;  
  ap_uint<32> LAYER_OUT_W_HW;
  // inst1
  ap_uint<32> LAYER_IN_NUM; 
  ap_uint<32> LAYER_OUT_NUM;
  ap_uint<32> LAYER_IN_H;
  ap_uint<32> LAYER_IN_W;  
  ap_uint<32> LAYER_OUT_H;  
  ap_uint<32> LAYER_OUT_W;  
  // inst2
  ap_uint<32> CIN_OFFSET; 
  ap_uint<32> WEIGHT_OFFSET;
  ap_uint<32> BIAS_OFFSET;
  ap_uint<32> COUT_OFFSET; 
  ap_uint<16> FILTER_S1; 
  ap_uint<16> FILTER_S2;   
  ap_uint<32> STRIDE;  
  // inst3
  ap_uint<32> LAYER_EN; 
  ap_uint<32> LAYER_IN_NUM_T;
  ap_uint<32> LAYER_OUT_NUM_T; 
  ap_uint<32> LAYER_IN_H_T; 
  ap_uint<32> LAYER_IN_W_T; 
  ap_uint<1>  DEPTH_CONV_EN;
  ap_uint<1>  CONV_EN;
  ap_uint<1>  RELU_EN;       
  ap_uint<1>  POOL_EN;       
  ap_uint<1>  UP_SAMPLE_EN;  
  ap_uint<1>  BIAS_EN;       
  ap_uint<1>  INTER_LOAD_EN;
  ap_uint<1>  INTER_WRITE_EN;

  // Read instructions
  ConfigInst inst0 = fifo_config_in.read();
  ConfigInst inst1 = fifo_config_in.read();
  ConfigInst inst2 = fifo_config_in.read();
  ConfigInst inst3 = fifo_config_in.read();
  ConfigInst inst4 = fifo_config_in.read();

  ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

  bool en_prev;
  bool max_pool_prev;
  uint LAYER_IN_NUM_prev;
  uint LAYER_OUT_NUM_prev;
  uint LAYER_IN_NUM_T_prev;
  uint LAYER_OUT_NUM_T_prev;
  uint LAYER_IN_H_T_prev;
  uint LAYER_IN_W_T_prev;
  uint LAYER_OUT_H_HW_prev;
  uint LAYER_OUT_W_HW_prev;
  uint cout_offset_prev;

  bool write_done = 0;
  uint task_cnt = 0;
  bool layer_start = 0;
  bool done = 0;
  // We assum that cin has been pre-padded with zeros
  while(!done){

    if (layer_start){
      inst0 = fifo_config_in.read();
      inst1 = fifo_config_in.read();
      inst2 = fifo_config_in.read();
      inst3 = fifo_config_in.read();
      inst4 = fifo_config_in.read();
      layer_start = 0;
    }

    // inst0
    LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
    LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
    LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
    LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
    LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
    LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
    // inst1
    LAYER_IN_NUM     = inst1(32*0+31, 32*0);
    LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
    LAYER_IN_H       = inst1(32*2+31, 32*2);
    LAYER_IN_W       = inst1(32*3+31, 32*3);
    LAYER_OUT_H      = inst1(32*4+31, 32*4);
    LAYER_OUT_W      = inst1(32*5+31, 32*5);
    // inst2
    CIN_OFFSET       = inst2(32*0+31, 32*0);
    WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
    BIAS_OFFSET      = inst2(32*2+31, 32*2);
    COUT_OFFSET      = inst2(32*3+31, 32*3);
    FILTER_S1        = inst2(32*4+15, 32*4);
    FILTER_S2        = inst2(32*4+31, 32*4+16);
    STRIDE           = inst2(32*5+31, 32*5);
    // inst3
    LAYER_EN         = inst3(32*0+31, 32*0);
    LAYER_IN_NUM_T   = inst3(32*1+31, 32*1);
    LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2);
    LAYER_IN_H_T     = inst3(32*3+31, 32*3);
    LAYER_IN_W_T     = inst3(32*4+31, 32*4);
  
    DEPTH_CONV_EN    = LAYER_EN[0];
    CONV_EN          = LAYER_EN[1];
    RELU_EN          = LAYER_EN[2];
    POOL_EN          = LAYER_EN[3];
    UP_SAMPLE_EN     = LAYER_EN[4]; // reserved
    BIAS_EN          = LAYER_EN[5];
    INTER_LOAD_EN    = LAYER_EN[6];
    INTER_WRITE_EN   = LAYER_EN[7];
  
    cout_offset = COUT_OFFSET;
    bool en = POOL_EN || (POOL_EN == 0 && STRIDE == 2);
    bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
    bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
    bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
  
    if (INTER_WRITE_EN == 0){

      if (task_cnt == 0){
        cout_write_fifo_read(
            cout_burst_buf_ping, fifo_cout, en, max_pool,
            LAYER_IN_NUM, LAYER_OUT_H, LAYER_OUT_W,
            LAYER_IN_NUM_T, LAYER_OUT_NUM_T,
            LAYER_IN_H_T, LAYER_IN_W_T,
            in_h_iter, in_w_iter
        );
      } else {
        if (task_cnt % 2 == 1){
          cout_write_fifo_read(
              cout_burst_buf_pong, fifo_cout, en, max_pool,
              LAYER_IN_NUM, LAYER_OUT_H, LAYER_OUT_W,
              LAYER_IN_NUM_T, LAYER_OUT_NUM_T, 
              LAYER_IN_H_T, LAYER_IN_W_T,
              in_h_iter, in_w_iter
          );
        
          cout_write_ddr_write(
            cout_burst_buf_ping, global_cout,
            en_prev, max_pool_prev,
            num_iter_prev, in_h_iter_prev, in_w_iter_prev,
            LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev, 
            LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev, 
            LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
            LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
            cout_offset_prev,
            !write_done
          );
  
        } else {
          cout_write_fifo_read(
              cout_burst_buf_ping, fifo_cout, en, max_pool,
              LAYER_IN_NUM, LAYER_OUT_H, LAYER_OUT_W,
              LAYER_IN_NUM_T, LAYER_OUT_NUM_T, 
              LAYER_IN_H_T, LAYER_IN_W_T,
              in_h_iter, in_w_iter
          );
  
          cout_write_ddr_write(
            cout_burst_buf_pong, global_cout,
            en_prev, max_pool_prev,
            num_iter_prev, in_h_iter_prev, in_w_iter_prev,
            LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev, 
            LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev, 
            LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
            LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
            cout_offset_prev,
            !write_done
          );
  
        }
      }
    
      if (task_cnt > 0){
        write_done = 1;
      }

      switch(en){
        case 0:
          {        
            task_cnt++;
            num_iter_prev = num_iter;
            in_h_iter_prev = in_h_iter;
            in_w_iter_prev = in_w_iter;
            en_prev = en;
            max_pool_prev = max_pool;
            LAYER_IN_NUM_prev = LAYER_IN_NUM;
            LAYER_OUT_NUM_prev = LAYER_OUT_NUM;
            LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
            LAYER_OUT_NUM_T_prev = LAYER_OUT_NUM_T;
            LAYER_IN_H_T_prev = LAYER_IN_H_T;
            LAYER_IN_W_T_prev = LAYER_IN_W_T;
            LAYER_OUT_H_HW_prev = LAYER_OUT_H_HW;
            LAYER_OUT_W_HW_prev = LAYER_OUT_W_HW;
            cout_offset_prev = cout_offset;
            write_done = 0;
          }
          break;
        case 1:
          {
            task_cnt++;
            num_iter_prev = num_iter;
            in_h_iter_prev = in_h_iter;
            in_w_iter_prev = in_w_iter;
            en_prev = en;
            max_pool_prev = max_pool;
            LAYER_IN_NUM_prev = LAYER_IN_NUM;
            LAYER_OUT_NUM_prev = LAYER_OUT_NUM;
            LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
            LAYER_OUT_NUM_T_prev = LAYER_OUT_NUM_T;
            LAYER_IN_H_T_prev = LAYER_IN_H_T;
            LAYER_IN_W_T_prev = LAYER_IN_W_T;
            LAYER_OUT_H_HW_prev = LAYER_OUT_H_HW;
            LAYER_OUT_W_HW_prev = LAYER_OUT_W_HW;
            cout_offset_prev = cout_offset;           
            write_done = 0;
          }
          break;
      }
    }

    if (max_pool){
      num_iter += LAYER_IN_NUM_T;
      if (num_iter >= LAYER_IN_NUM){
        num_iter = 0;
        in_h_iter += LAYER_IN_H_T;
        if (in_h_iter >= LAYER_IN_H){
          in_h_iter = 0;
          in_w_iter += LAYER_IN_W_T;
          if (in_w_iter >= LAYER_IN_W){
            in_w_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    } else {
      in_h_iter += LAYER_IN_H_T;
      if (in_h_iter >= LAYER_IN_H){
        in_h_iter = 0;
        in_w_iter += LAYER_IN_W_T;
        if (in_w_iter >= LAYER_IN_W){
          in_w_iter = 0;
          num_iter += LAYER_OUT_NUM_T;
          if (num_iter >= LAYER_OUT_NUM){
            num_iter = 0;
            layer_iter += 1;
            layer_start = 1;
            if (layer_iter == LAYER_BATCH){              
              layer_iter = 0;
              done = 1;
            }
          }
        }
      }
    }

  }

  if (INTER_WRITE_EN == 0){
    if (task_cnt % 2 == 1){
      cout_write_ddr_write(
        cout_burst_buf_ping, global_cout,
        en_prev, max_pool_prev,
        num_iter_prev, in_h_iter_prev, in_w_iter_prev,
        LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev, 
        LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev, 
        LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
        LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
        cout_offset_prev,
        !write_done
      );
    } else {
      cout_write_ddr_write(
        cout_burst_buf_pong, global_cout,
        en_prev, max_pool_prev,
        num_iter_prev, in_h_iter_prev, in_w_iter_prev,
        LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev, 
        LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev, 
        LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
        LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
        cout_offset_prev,
        !write_done
      );
    }
  }
}

void engine(
  bus_t0 *global_cin,
  bus_t1 *global_weight,
  bus_t2 *global_bias,
  bus_t0 *global_cout,
  uint    config[CONFIG_PARAMS]
){
#pragma HLS DATAFLOW
  /**
   * Comments: naming rule
   * Datatypes: <module_name> + 'Data' + <port_number> + 'Type'
   * Instnames: fifo + <module_name> + <port_number>
  */  
  // --------------------------------------------
  // Definitions of data fifos
  // --------------------------------------------
  // Module: cin_load 
  // Output ports: 
  // 0: cout -> inter_load
  hls::stream<CinLoadData0Type> fifo_cin_load_0;
#pragma HLS STREAM variable=fifo_cin_load_0 depth=16
  // Module: weight_load
  // Output ports:
  // 0: weight -> depth_conv
  // 1: weight -> conv
  // 2: weight -> relu
  hls::stream<WeightLoadData0Type> fifo_weight_load_0;
  hls::stream<WeightLoadData1Type> fifo_weight_load_1;
  hls::stream<WeightLoadData2Type> fifo_weight_load_2;
#pragma HLS STREAM variable=fifo_weight_load_0 depth=64
#pragma HLS STREAM variable=fifo_weight_load_1 depth=64
#pragma HLS STREAM variable=fifo_weight_load_2 depth=64
  // Module: inter_load
  // Output ports:
  // 0: cout -> depth_conv
  hls::stream<InterLoadData0Type> fifo_inter_load_0;
#pragma HLS STREAM variable=fifo_inter_load_0 depth=16
  // Module: depth_conv
  // Output ports:
  // 0: cout -> conv
  hls::stream<DepthConvData0Type> fifo_depth_conv_0;
#pragma HLS STREAM variable=fifo_depth_conv_0 depth=16
  // Module: conv
  // Output ports:
  // 0: cout -> relu
  hls::stream<ConvData0Type> fifo_conv_0;
#pragma HLS STREAM variable=fifo_conv_0 depth=16
  // Module: relu
  // Output ports:
  // 0: cout -> pool
  hls::stream<ReluData0Type> fifo_relu_0;
#pragma HLS STREAM variable=fifo_relu_0 depth=16
  // Module: pool
  // Output ports:
  // 0: cout -> inter_write
  hls::stream<PoolData0Type> fifo_pool_0;
#pragma HLS STREAM variable=fifo_pool_0 depth=16
  // Module: inter_write
  // Output ports:
  // 0: cout -> cout_write
  // 1: cout -> inter_load
  hls::stream<InterWriteData0Type> fifo_inter_write_0;
  static hls::stream<InterWriteData1Type> fifo_inter_write_1;
#pragma HLS STREAM variable=fifo_inter_write_0 depth=16
#pragma HLS STREAM variable=fifo_inter_write_1 depth=16
  // Module: upsample
  // Output ports:
  // 0: cout -> data_write
//  hls::stream<UpsampleData0Type> fifo_upsample_0;
//#pragma HLS STREAM variable=fifo_upsample_0 depth=64
  // Module: cout_write
  // Ouput ports:
  // 0: cout -> global_cout

  // ----------------------------------------------
  // Definitions of config fifos
  // ----------------------------------------------
  hls::stream<ConfigInst> config_weight_load;
  hls::stream<ConfigInst> config_inter_load;
  hls::stream<ConfigInst> config_depth_conv;
  hls::stream<ConfigInst> config_conv;
  hls::stream<ConfigInst> config_relu;
  hls::stream<ConfigInst> config_pool;
  hls::stream<ConfigInst> config_upsample; // reserved
  hls::stream<ConfigInst> config_inter_write;
  hls::stream<ConfigInst> config_data_write;
#pragma HLS STREAM variable=config_weight_load depth=16
#pragma HLS STREAM variable=config_inter_load depth=16
#pragma HLS STREAM variable=config_depth_conv depth=16
#pragma HLS STREAM variable=config_conv depth=16
#pragma HLS STREAM variable=config_relu depth=16
#pragma HLS STREAM variable=config_pool depth=16
#pragma HLS STREAM variable=config_upsample depth=16
#pragma HLS STREAM variable=config_inter_write depth=16
#pragma HLS STREAM variable=config_data_write depth=16

  cin_load(
    global_cin, // global_weight, global_bias,
    config,
    fifo_cin_load_0, // fifo_data_load_1, fifo_data_load_2, fifo_data_load_3,
    config_weight_load
    );
  weight_load(
    global_weight, global_bias,  
    config_weight_load,
    fifo_weight_load_0, fifo_weight_load_1, fifo_weight_load_2,
    config_inter_load
    );
  inter_load(
    fifo_cin_load_0,
    fifo_inter_write_1,
    config_inter_load,
    fifo_inter_load_0,
    config_depth_conv
    );
  depth_conv(
//    fifo_cin_load_0, 
    fifo_inter_load_0,
    fifo_weight_load_0,
    config_depth_conv,
    fifo_depth_conv_0, 
    config_conv
    );  
  conv(
    fifo_depth_conv_0, fifo_weight_load_1,
    config_conv,
    fifo_conv_0,
    config_relu
    );
  relu(
    fifo_conv_0, fifo_weight_load_2,
    config_relu,
    fifo_relu_0,
    config_pool
    );
  pool(
    fifo_relu_0,
    config_pool,
    fifo_pool_0,
//    config_data_write
    config_inter_write
    );
  inter_write(
    fifo_pool_0,
    config_inter_write,
    fifo_inter_write_0,
//    fifo_inter_write_1, // reserved
    config_data_write
    );
//  upsample(
//    fifo_pool_0,
//    config_upsample,
//    fifo_upsample_0,
//    config_data_write
//    );
  cout_write(
    fifo_inter_write_0,
//    fifo_pool_0,
    config_data_write,
    global_cout
    );

}

void top_kernel(
  bus_t0 *global_cin,
  bus_t0 *global_cout,
  bus_t1 *global_weight,
  bus_t2 *global_bias,
  bus_t3 *layer_config
){
#pragma HLS INTERFACE m_axi port=global_cin offset=slave bundle=gmem1 depth=0
#pragma HLS INTERFACE m_axi port=global_cout offset=slave bundle=gmem1 depth=29982
#pragma HLS INTERFACE m_axi port=global_weight offset=slave bundle=gmem2 depth=131409
#pragma HLS INTERFACE m_axi port=global_bias offset=slave bundle=gmem2 depth=473
#pragma HLS INTERFACE m_axi port=layer_config offset=slave bundle=gcontrol depth=2267

#pragma HLS INTERFACE s_axilite port=global_cin bundle=control
#pragma HLS INTERFACE s_axilite port=global_weight bundle=control
#pragma HLS INTERFACE s_axilite port=global_bias bundle=control
#pragma HLS INTERFACE s_axilite port=global_cout bundle=control
#pragma HLS INTERFACE s_axilite port=layer_config bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
 
  // Copy the first instruction
  unsigned int init_inst[4]; // [VGG_LAYERS, STAGE1_LAYERS, STAGE2_LAYRES, STAGE2_ITER]
  memcpy((void*)init_inst, (void*)(&layer_config[0]), sizeof(unsigned int) * 4);
  int vgg_layers = init_inst[0];
  int stage1_layers = init_inst[1];
  int stage2_layers = init_inst[2];
  int stage2_iter = init_inst[3];

  int layer_num = vgg_layers + stage1_layers * 2 + stage2_layers * 2 * stage2_iter;

  unsigned int config[CONFIG_PARAMS * MAX_LAYER_BATCH]; 
  int cur_layer_batch = 1;
  int nxt_layer_batch = 1;
  int layer_id = 0;
  while(layer_id < layer_num){   
    cur_layer_batch = nxt_layer_batch;
//    cout << layer_id << " " << cur_layer_batch << endl;
    memcpy((void*)config, (void*)(&layer_config[4 + CONFIG_PARAMS * layer_id]), sizeof(unsigned int) * CONFIG_PARAMS * cur_layer_batch);
    nxt_layer_batch = config[CONFIG_PARAMS * (cur_layer_batch - 1) + 25 - 1];
    config[25 - 1] = cur_layer_batch;
    
    engine(global_cin, global_weight, global_bias, global_cout, config);
    layer_id += cur_layer_batch;
  }
}
