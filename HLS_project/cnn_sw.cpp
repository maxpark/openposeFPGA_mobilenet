#include "pose.h"

void extract_layer(
  data_t0* cin_hw,
  uint*    configs,
  uint     layer_id
){
  uint config[CONFIG_PARAMS];
  for (int p = 0; p < CONFIG_PARAMS; p++){
    config[p] = configs[4 + layer_id * CONFIG_PARAMS + p];
  }
  
  // inst0
  ap_uint<32> LAYER_IN_NUM_HW  = config[0];
  ap_uint<32> LAYER_OUT_NUM_HW = config[1];
  ap_uint<32> LAYER_IN_H_HW    = config[2];
  ap_uint<32> LAYER_IN_W_HW    = config[3];
  ap_uint<32> LAYER_OUT_H_HW   = config[4];
  ap_uint<32> LAYER_OUT_W_HW   = config[5];

  // inst1
  ap_uint<32> LAYER_IN_NUM  = config[6];
  ap_uint<32> LAYER_OUT_NUM = config[7];
  ap_uint<32> LAYER_IN_H    = config[8];
  ap_uint<32> LAYER_IN_W    = config[9];
  ap_uint<32> LAYER_OUT_H   = config[10];
  ap_uint<32> LAYER_OUT_W   = config[11];
 
  // inst2
  ap_uint<32> CIN_OFFSET    = config[12];
  ap_uint<32> WEIGHT_OFFSET = config[13];
  ap_uint<32> BIAS_OFFSET   = config[14];
  ap_uint<32> COUT_OFFSET   = config[15];
  ap_uint<16> FILTER_S1     = config[16];
  ap_uint<16> FILTER_S2     = config[17];
  ap_uint<32> STRIDE        = config[18];

  // inst3
  ap_uint<32> LAYER_EN        = config[19];
  ap_uint<32> LAYER_IN_NUM_T  = config[20];
  ap_uint<32> LAYER_OUT_NUM_T = config[21]; 
  ap_uint<32> LAYER_IN_H_T    = config[22]; 
  ap_uint<32> LAYER_IN_W_T    = config[23]; 

  ap_uint<1>  DEPTH_CONV_EN = LAYER_EN[0];
  ap_uint<1>  CONV_EN       = LAYER_EN[1];
  ap_uint<1>  RELU_EN       = LAYER_EN[2];
  ap_uint<1>  POOL_EN       = LAYER_EN[3];
  ap_uint<1>  UP_SAMPLE_EN  = LAYER_EN[4];  // reserved
  ap_uint<1>  LD_SEL        = LAYER_EN[5];  // reserved
  ap_uint<1>  WR_SEL        = LAYER_EN[6];  // reserved

  data_t0* layer_cout = new data_t0[LAYER_OUT_NUM * LAYER_OUT_H * LAYER_OUT_W];
  uint cout_offset = COUT_OFFSET;

  // extract input
  ofstream input_file("debug_input.dat");
  if (input_file.is_open()){
    for (int o1 = 0; o1 < LAYER_IN_NUM_HW / LAYER_IN_NUM_T; o1++)
      for (int h = 0; h < LAYER_IN_H_T + FILTER_S1 - 1; h++)
        for (int w = 0; w < LAYER_IN_W_T + FILTER_S1 - 1; w++)
          for (int o2 = 0; o2 < LAYER_IN_NUM_T; o2++){
            uint o = o1 * LAYER_IN_NUM_T + o2;
            uint global_cin_idx = o1 * LAYER_IN_H_HW * LAYER_IN_W_HW * LAYER_IN_NUM_T + h * LAYER_IN_W_HW * LAYER_IN_NUM_T + w * LAYER_IN_NUM_T + o2 + CIN_OFFSET;
            if (o == 0){
              input_file << cin_hw[global_cin_idx] << endl;
            }
          }
    input_file.close();
  }

  // write out to files
  cout << "extract offset: " << cout_offset << endl;
  cout << cout_offset + 1 << ": " << cin_hw[cout_offset + 1] << endl;
  cout << cout_offset + 32 << ": " << cin_hw[cout_offset + 32] << endl;
  cout << LAYER_OUT_NUM_HW << " " << LAYER_OUT_NUM_T << " " << endl;
  ofstream output_file("debug.dat");
  if (output_file.is_open()){
    for (int o1 = 0; o1 < LAYER_OUT_NUM_HW / LAYER_OUT_NUM_T; o1++)
      for (int h = 0; h < LAYER_OUT_H; h++)
        for (int w = 0; w < LAYER_OUT_W; w++)
          for (int o2 = 0; o2 < LAYER_OUT_NUM_T; o2++){
            uint o = o1 * LAYER_OUT_NUM_T + o2;
            uint global_cout_idx = o1 * LAYER_OUT_H_HW * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + 
              h * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + w * LAYER_OUT_NUM_T + o2 + cout_offset;
            uint local_cout_idx = o * LAYER_OUT_H * LAYER_OUT_W + h * LAYER_OUT_W + w;
#ifdef DEBUG            
            if (h == 0 && w == 1 && o == 0){
              cout << global_cout_idx << " " << local_cout_idx << endl;
            }
#endif              
            if (o < LAYER_OUT_NUM){
              layer_cout[local_cout_idx] = cin_hw[global_cout_idx];
            }
          }

    cout << "extract_layer: " << LAYER_OUT_H << " " << LAYER_OUT_W << " " << LAYER_OUT_NUM << endl;

    for (int h = 0; h < LAYER_OUT_H; h++)
      for (int w = 0; w < LAYER_OUT_W; w++)
        for (int o = 0; o < LAYER_OUT_NUM; o++){
          uint local_cout_idx = o * LAYER_OUT_H * LAYER_OUT_W + h * LAYER_OUT_W + w;
          output_file << layer_cout[local_cout_idx] << endl;
        }
  } else {
    cout << "Output open failed!" << endl;
    exit(-1);
  }

  delete[] layer_cout;
}

// Loads inputs, weights, and bias data
void openpose_preprocess(
  data_t0* cin_hw,
  data_t1* weight_hw,
  data_t2* bias_hw,
  data_t0  LAYER_out[STAGE2L_OUT_H][STAGE2L_OUT_W][STAGE2R_OUT_NUM + STAGE2L_OUT_NUM]
//  data_t0  LAYERL_out[STAGE2L_OUT_NUM][STAGE2L_OUT_H][STAGE2L_OUT_W],  
//  data_t0  LAYERR_out[STAGE2R_OUT_NUM][STAGE2R_OUT_H][STAGE2R_OUT_W]
){
  char* prj_path_c = getenv("PRJ_PATH");  

  // Prepare the software buffers
  cout << std::fixed << "Preparing data..." << endl;
  
  // Load the inputs for the network
  static data_t0 LAYER1_cin[LAYER1_IN_NUM][LAYER1_IN_H][LAYER1_IN_W];
  cout << "Loading input..." << endl; 
  string file_path = string(prj_path_c) + "/data/input.bin";  
  ifstream input_file(file_path.c_str(), ios::binary | ios::in);
  char* bin_input = new char[sizeof(data_t0) * LAYER1_IN_NUM * LAYER1_IN_H *LAYER1_IN_W];
  if (input_file.is_open()){
    input_file.read(bin_input, sizeof(data_t0) * LAYER1_IN_NUM * LAYER1_IN_H * LAYER1_IN_W);
    data_t0* convt_input = (data_t0*)bin_input;

    int idx = 0;
    for (int h = 0; h < LAYER1_IN_H; h++)
      for (int w = 0; w < LAYER1_IN_W; w++)
        for (int i = 0; i < LAYER1_IN_NUM; i++){
          LAYER1_cin[i][h][w] = convt_input[idx];
          idx++;
        }

    input_file.close();
  } else {
    cout << "Input open failed!" << endl;
    exit(-1);
  }
  delete[] bin_input;

  // Initialize the hardware input buffer
  // Cin layout: [IN_NUM / IN_NUM_T][IN_H + K - 1][IN_W + K - 1][IN_NUM_T]
  for (int i1 = 0; i1 < LAYER1_IN_NUM_HW / LAYER1_IN_NUM_T; i1++)
    for (int h = 0; h < LAYER1_IN_H; h++)
      for (int w = 0; w < LAYER1_IN_W; w++)
        for (int i2 = 0; i2 < LAYER1_IN_NUM_T; i2++){
          int i = i1 * LAYER1_IN_NUM_T + i2;
          if (i < LAYER1_IN_NUM){
            cin_hw[i1 * LAYER1_IN_H_HW * LAYER1_IN_W_HW * LAYER1_IN_NUM_T + (h + int(LAYER1_K / 2)) * LAYER1_IN_W_HW * LAYER1_IN_NUM_T + (w + int(LAYER1_K / 2)) * LAYER1_IN_NUM_T + i2] = LAYER1_cin[i][h][w]; // filter size = 3
          }
        }

  // Load weights
  cout << "Loading weight..." << endl;
  file_path = string(prj_path_c) + "/data/weight_reorg.bin";  
  ifstream weight_file(file_path.c_str(), ios::binary | ios::in);
  bin_input = new char[sizeof(data_t1) * WEIGHT_SIZE];
  if (weight_file.is_open()){
    weight_file.read(bin_input, sizeof(data_t1) * WEIGHT_SIZE);
    data_t1* convt_input = (data_t1*)bin_input;

    for (int w = 0; w < WEIGHT_SIZE; w++){
      weight_hw[w] = convt_input[w];
    }

    weight_file.close();
  } else {
    cout << "Weight open failed!" << endl;
    exit(-1);
  }
  
  delete[] bin_input;

  // Load bias
  cout << "Loading bias..." << endl;
  file_path = string(prj_path_c) +  "/data/bias_reorg.bin";
  ifstream bias_file(file_path.c_str(), ios::binary | ios::in);
  bin_input = new char[sizeof(data_t2) * BIAS_SIZE];  

  if (bias_file.is_open()){
    bias_file.read(bin_input, sizeof(data_t2) * BIAS_SIZE);
    data_t2* convt_input = (data_t2*)bin_input;

    for (int w = 0; w < BIAS_SIZE; w++){
      bias_hw[w] = convt_input[w];    
    }
    bias_file.close();
  } else {
    cout << "Bias open failed!" << endl;
    exit(-1);
  }

  delete[] bin_input;

  // Load outputs
  cout << "Loading output..." << endl;
  file_path = string(prj_path_c) + "/data/output.bin";
  ifstream output_file(file_path.c_str(), ios::binary | ios::in);
  bin_input = new char[sizeof(data_t0) * (STAGE2L_OUT_H * STAGE2L_OUT_W * STAGE2L_OUT_NUM + 
      STAGE2R_OUT_H * STAGE2R_OUT_W * STAGE2R_OUT_NUM)];  

  if (output_file.is_open()){
    output_file.read(bin_input, sizeof(data_t0) * (STAGE2L_OUT_H * STAGE2L_OUT_W * STAGE2L_OUT_NUM + STAGE2R_OUT_H * STAGE2R_OUT_W * STAGE2R_OUT_NUM));
    data_t0* convt_input = (data_t0*)bin_input;

    int idx = 0;
    for (int h = 0; h < STAGE2R_OUT_H; h++)
      for (int w = 0; w < STAGE2R_OUT_W; w++)
        for (int o = 0; o < STAGE2R_OUT_NUM + STAGE2L_OUT_NUM; o++){
          LAYER_out[h][w][o] = convt_input[idx];
          idx++;
        }
    output_file.close();
  } else {
    cout << "Output open failed!" << endl;
    exit(-1);
  }

  delete[] bin_input;
}

// Extract hardware outputs
void openpose_postprocess(
  data_t0* cin_hw,
  data_t0  LAYER_out[STAGE2L_OUT_H][STAGE2L_OUT_W][STAGE2R_OUT_NUM + STAGE2L_OUT_NUM]
//  data_t0 LAYERL_out[STAGE2L_OUT_NUM][STAGE2L_OUT_H][STAGE2L_OUT_W],  
//  data_t0 LAYERR_out[STAGE2R_OUT_NUM][STAGE2R_OUT_H][STAGE2R_OUT_W]
){
  // Cout layout: [OUT_NUM / OUT_NUM_T][OUT_H + K - 1][OUT_W + K - 1][OUT_NUM_T]
  for (int o1 = 0; o1 < STAGE2L_OUT_NUM_HW / STAGE2L_OUT_NUM_T; o1++)
    for (int h = 0; h < STAGE2L_OUT_H; h++)
      for (int w = 0; w < STAGE2L_OUT_W; w++)
        for (int o2 = 0; o2 < STAGE2L_OUT_NUM_T; o2++){
          int o = o1 * STAGE2L_OUT_NUM_T + o2;
          if (o < STAGE2L_OUT_NUM){
            LAYER_out[h][w][o + STAGE2R_OUT_NUM] = cin_hw[STAGE2L_OFFSET + o1 * STAGE2L_OUT_H_HW * STAGE2L_OUT_W_HW * STAGE2L_OUT_NUM_T + (h + int(STAGE2L_K / 2)) * STAGE2L_OUT_W_HW * STAGE2L_OUT_NUM_T + (w + int(STAGE2L_K / 2)) * STAGE2L_OUT_NUM_T + o2];
          }
        }
  for (int o1 = 0; o1 < STAGE2R_OUT_NUM_HW / STAGE2R_OUT_NUM_T; o1++)
    for (int h = 0; h < STAGE2R_OUT_H; h++)
      for (int w = 0; w < STAGE2R_OUT_W; w++)
        for (int o2 = 0; o2 < STAGE2R_OUT_NUM_T; o2++){
          int o = o1 * STAGE2R_OUT_NUM_T + o2;
          if (o < STAGE2R_OUT_NUM){
            LAYER_out[h][w][o] = cin_hw[STAGE2R_OFFSET + o1 * STAGE2R_OUT_H_HW * STAGE2R_OUT_W_HW * STAGE2R_OUT_NUM_T + (h + int(STAGE2R_K / 2)) * STAGE2R_OUT_W_HW * STAGE2R_OUT_NUM_T + (w + int(STAGE2R_K / 2)) * STAGE2R_OUT_NUM_T + o2];
          }
        }
}
