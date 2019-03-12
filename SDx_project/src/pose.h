#ifndef __POSE_H_
#define __POSE_H_

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstring>

#include "ap_fixed.h"
#include "hls_stream.h"
#include "ap_int.h"

#include "params.h"

//#define DEBUG

using namespace std;
using namespace hls;

// Macros
/*
 * Set the module lanes to the same number currently
 */
#define SIMD_LANE 8
#define DATA_LDWR_LANE (SIMD_LANE)
#define DEPTH_CONV_LANE (SIMD_LANE)
#define CONV_LANE (SIMD_LANE)
#define RELU_LANE (SIMD_LANE)
#define POOL_LANE (SIMD_LANE)
#define UPSAMPLE_LANE (SIMD_LANE) // reserved
#define INTER_LOAD_LANE (SIMD_LANE)
#define INTER_WRITE_LANE (SIMD_LANE)
//#define DATA_WRITE_LANE 8

#define CONFIG_PARAMS  31
#define INST_PER_LAYER 5

// Data Types
// primtive data types
typedef float data_t0; // cin, cout
typedef float data_t1; // weight
typedef float data_t2; // bias
typedef unsigned int data_t3; // inst
#define DATA_W0 32
#define DATA_W1 32
#define DATA_W2 32
#define DATA_W3 32
typedef ap_uint<512> bus_t0;
typedef ap_uint<512> bus_t1;
typedef ap_uint<512> bus_t2;
typedef ap_uint<32> bus_t3;
#define BUS_PACK_FACTOR0 (512 / DATA_W0)
#define BUS_PACK_FACTOR1 (512 / DATA_W1)
#define BUS_PACK_FACTOR2 (512 / DATA_W2)
#define BUS_PACK_FACTOR3 (32 / DATA_W3)

#define DATA_SEL_FACTOR0 (BUS_PACK_FACTOR0 / SIMD_LANE)
#define DATA_SEL_FACTOR1 (BUS_PACK_FACTOR1 / SIMD_LANE)
#define DATA_SEL_FACTOR2 (BUS_PACK_FACTOR2 / SIMD_LANE)

typedef unsigned int uint;
typedef ap_uint<192> ConfigInst;
// Inst0: in_num_hw  | out_num_hw    | in_h_hw     | in_w_hw     | out_h_hw | out_w_hw
// Inst1: in_num     | out_num       | in_h        | in_w        | out_h    | out_w
// Inst2: cin_offset | weight_offset | bias_offset | cout_offset | filter_s1, filter_s2 | stride
// Inst3: layer_en: depth_conv_en, conv_en, relu_en, pool_en, up_sample_en, bias_en, inter_load_en, inter_write_en | in_num_t | out_num_t | in_h_t | in_w_t | nxt_layer_batch
// Inst4: task_num1 | task_num2 | local_accum_num | local_reg_num | row_il_factor | col_il_factor

typedef ap_uint<DATA_W0 * DEPTH_CONV_LANE> CinLoadData0Type;

typedef ap_uint<DATA_W1 * DEPTH_CONV_LANE> WeightLoadData0Type;
typedef ap_uint<DATA_W1 * CONV_LANE> WeightLoadData1Type;
typedef ap_uint<DATA_W1 * RELU_LANE> WeightLoadData2Type;

typedef ap_uint<DATA_W0 * INTER_LOAD_LANE> InterLoadData0Type;
typedef ap_uint<DATA_W0 * CONV_LANE> DepthConvData0Type;
typedef ap_uint<DATA_W0 * RELU_LANE> ConvData0Type;
typedef ap_uint<DATA_W0 * POOL_LANE> ReluData0Type;
typedef ap_uint<DATA_W0 * DATA_LDWR_LANE> PoolData0Type; 
typedef ap_uint<DATA_W0 * INTER_WRITE_LANE> InterWriteData0Type;
typedef ap_uint<DATA_W0 * INTER_WRITE_LANE> InterWriteData1Type;
// typedef ap_uint<CIN_DATA_W*DATA_WRITE_LANE> UpsampleData0Type;

extern "C" {
void top_kernel(
  bus_t0 *global_cin,
  bus_t0 *global_cout,
  bus_t1 *global_weight,
  bus_t2 *global_bias,
  bus_t3 *layer_config
);
}

void openpose_preprocess(
  data_t0* cin_hw,
  data_t1* weight_hw,
  data_t2* bias_hw,
  data_t0  LAYER_out[STAGE2L_OUT_H][STAGE2L_OUT_W][STAGE2R_OUT_NUM + STAGE2L_OUT_NUM]
//  data_t0 LAYERL_out[STAGE2L_OUT_NUM][STAGE2L_OUT_H][STAGE2L_OUT_W],  
//  data_t0 LAYERR_out[STAGE2R_OUT_NUM][STAGE2R_OUT_H][STAGE2R_OUT_W]
);

void instInit(uint config[LAYER_NUM*CONFIG_PARAMS]);

void openpose_postprocess(
  data_t0* cin_hw,
  data_t0  LAYER_out[STAGE2L_OUT_H][STAGE2L_OUT_W][STAGE2R_OUT_NUM + STAGE2L_OUT_NUM]
//  data_t0 LAYERL_out[STAGE2L_OUT_NUM][STAGE2L_OUT_H][STAGE2L_OUT_W],  
//  data_t0 LAYERR_out[STAGE2R_OUT_NUM][STAGE2R_OUT_H][STAGE2R_OUT_W]
);

void extract_layer(
  data_t0* cin_hw,
  uint*    config,
  uint     layer_id
);

void compute_layer();

/**
 * Helper Functions
 * - max
 * - reinterpret
 * - stencil_w3
 * - stencil_w1
 * - maxpool_w2 
 */
#define max(a,b) ((a)>(b)?(a):(b))

// systolic array kernel
void kernel(
  stream<DepthConvData0Type>  &fifo_cin,
  stream<WeightLoadData1Type> &fifo_weight,
  stream<ConvData0Type>       &fifo_cout,
  stream<ConfigInst>          &fifo_config_in,
  stream<ConfigInst>          &fifo_config_out
);

template <typename To, typename From>
inline To Reinterpret(const From& val){
  return reinterpret_cast<const To&>(val);
}

template <class T_data_t0, class T_data_t1, int T_IN_NUM_T, int T_IN_H_T, int T_IN_W_T, int T_UNROLL, int T_WS, int T_DATA_WIDTH0, int T_DATA_WIDTH1>
void stencil_w3(
  hls::stream<ap_uint<T_DATA_WIDTH0 * T_UNROLL> > &fifo_in,
  T_data_t1                                       weights[T_IN_NUM_T / T_UNROLL][T_UNROLL][K_T][K_T],  
  hls::stream<ap_uint<T_DATA_WIDTH0 * T_UNROLL> > &fifo_out,
  uint                                            stride,
  uint                                            layer_in_num_t,
  uint                                            layer_in_h_t
){
#pragma HLS INLINE off
  T_data_t0 line_buf1[T_UNROLL][T_IN_W_T];
  T_data_t0 line_buf2[T_UNROLL][T_IN_W_T];
  T_data_t0 line_buf3[T_UNROLL][T_WS];
#pragma HLS ARRAY_PARTITION variable=line_buf1 dim=1 complete
#pragma HLS ARRAY_PARTITIOn variable=line_buf1 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=line_buf2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=line_buf2 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=line_buf3 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=line_buf3 dim=2 complete

  bool col_skip = 0;
  bool row_skip = 0;
  bool col_strip_skip = 0;
  bool row_strip_skip = 0;
  uint trans_cnt = 0;
  uint inner_trans_cnt = 0;

  ap_uint<T_DATA_WIDTH0> utmp[T_UNROLL];
#pragma HLS ARRAY_PARTITION variable=utmp complete
  T_data_t0 sums[T_UNROLL];
#pragma HLS ARRAY_PARTITION variable=sums complete

  int oo = 0;
  int iter = 0;
  int oo_bound = layer_in_num_t / T_UNROLL;
  int iter_bound = layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1;
  int bound = oo_bound * iter_bound;

//  for (int oo = 0; oo < layer_in_num_t / T_UNROLL; oo++){
//    uint trans_cnt = 0;
//    uint inner_trans_cnt = 0;

//  for (int iter = 0; iter < layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1; iter++){
  for (int total_iter = 0; total_iter < bound; total_iter++){
#pragma HLS PIPELINE II=1    
    if (iter == 0){
      trans_cnt = 0;
      inner_trans_cnt = 0;
    }

    ap_uint<T_DATA_WIDTH0 * T_UNROLL> wide_data_in;
//    if (iter < T_IN_H_T * T_IN_W_T){
    if (iter < layer_in_h_t * T_IN_W_T){
      wide_data_in = fifo_in.read();
    }

    for (int dup = 0; dup < T_UNROLL; dup++){
      T_data_t0 tmp1 = line_buf1[dup][T_IN_W_T - 1];
      T_data_t0 tmp2 = line_buf2[dup][T_IN_W_T - 1];
      for (int i = T_IN_W_T - 1; i >= 1; i--){
#pragma HLS UNROLL
        line_buf1[dup][i] = line_buf1[dup][i - 1];
        line_buf2[dup][i] = line_buf2[dup][i - 1];
//        line_buf3[dup][i] = line_buf3[dup][i - 1];
      }
      for (int i = T_WS - 1; i >= 1; i--){
#pragma HLS UNROLL
        line_buf3[dup][i] = line_buf3[dup][i - 1];
      }
      
//      if (iter < T_IN_H_T * T_IN_W_T){        
      if (iter < layer_in_h_t * T_IN_W_T){        
        ap_uint<T_DATA_WIDTH0> sel_tmp;
#if DEPTH_CONV_LANE == 16
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
          case 4:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 5 - 1, T_DATA_WIDTH0 * 4);
            break;
          case 5:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 6 - 1, T_DATA_WIDTH0 * 5);
            break;
          case 6:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 7 - 1, T_DATA_WIDTH0 * 6);
            break;
          case 7:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 8 - 1, T_DATA_WIDTH0 * 7);
            break;
          case 8:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 9 - 1, T_DATA_WIDTH0 * 8);
            break;
          case 9:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 10 - 1, T_DATA_WIDTH0 * 9);
            break;
          case 10:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 11 - 1, T_DATA_WIDTH0 * 10);
            break;
          case 11:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 12 - 1, T_DATA_WIDTH0 * 11);
            break;
          case 12:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 13 - 1, T_DATA_WIDTH0 * 12);
            break;
          case 13:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 14 - 1, T_DATA_WIDTH0 * 13);
            break;
          case 14:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 15 - 1, T_DATA_WIDTH0 * 14);
            break;
          case 15:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 16 - 1, T_DATA_WIDTH0 * 15);
            break;
        }
#elif DEPTH_CONV_LANE == 8
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
          case 4:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 5 - 1, T_DATA_WIDTH0 * 4);
            break;
          case 5:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 6 - 1, T_DATA_WIDTH0 * 5);
            break;
          case 6:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 7 - 1, T_DATA_WIDTH0 * 6);
            break;
          case 7:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 8 - 1, T_DATA_WIDTH0 * 7);
            break;
        }
#elif DEPTH_CONV_LANE == 4
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
        }
#elif DEPTH_CONV_LANE == 2
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
        }
#elif DEPTH_CONV_LANE == 1
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
        }
#endif        
        line_buf1[dup][0] = Reinterpret<T_data_t0>(sel_tmp);
#ifdef DEBUG
//        cout << "stencil_w3: " << line_buf1[dup][0] << " iter: " << iter << endl;
#endif        
//        line_buf1[dup][0] = data_select<data_t, ap_uint<DATA_WIDTH * UNROLL>, DATA_WIDTH>(wide_data_in, dup);        
      } else {      
        line_buf1[dup][0] = 0.0;
      }
      line_buf2[dup][0] = tmp1;
      line_buf3[dup][0] = tmp2;

      // mults
      T_data_t0 prod_0_0 = line_buf3[dup][T_WS - 1] * weights[oo][dup][0][0];
      T_data_t0 prod_0_1 = line_buf3[dup][T_WS - 2] * weights[oo][dup][0][1];
      T_data_t0 prod_0_2 = line_buf3[dup][T_WS - 3] * weights[oo][dup][0][2];
      T_data_t0 prod_1_0 = line_buf2[dup][T_WS - 1] * weights[oo][dup][1][0];
      T_data_t0 prod_1_1 = line_buf2[dup][T_WS - 2] * weights[oo][dup][1][1];
      T_data_t0 prod_1_2 = line_buf2[dup][T_WS - 3] * weights[oo][dup][1][2];
      T_data_t0 prod_2_0 = line_buf1[dup][T_WS - 1] * weights[oo][dup][2][0];
      T_data_t0 prod_2_1 = line_buf1[dup][T_WS - 2] * weights[oo][dup][2][1];
      T_data_t0 prod_2_2 = line_buf1[dup][T_WS - 3] * weights[oo][dup][2][2];
    
      // adds
      T_data_t0 sum_0_0 = prod_0_0 + prod_0_1;
      T_data_t0 sum_0_1 = prod_0_2 + prod_1_0;
      T_data_t0 sum_0_2 = prod_1_1 + prod_1_2;
      T_data_t0 sum_0_3 = prod_2_0 + prod_2_1;
      T_data_t0 sum_1_0 = sum_0_0 + sum_0_1;
      T_data_t0 sum_1_1 = sum_0_2 + sum_0_3;
      T_data_t0 sum_2_0 = sum_1_0 + sum_1_1;
      T_data_t0 sum_3_0 = sum_2_0 + prod_2_2;

      sums[dup]  = sum_3_0;
    }

    if (iter >= (T_WS - 1) * T_IN_W_T + T_WS - 1){      
      col_skip = (inner_trans_cnt % stride != stride - 1);
      row_skip = ((inner_trans_cnt / (T_IN_W_T - (T_WS - 1))) % stride != stride - 1);
      col_strip_skip = trans_cnt % T_IN_W_T >= (T_IN_W_T - (T_WS - 1));
//      row_strip_skip = trans_cnt / T_IN_W_T >= (T_IN_H_T - (T_WS - 1));
      row_strip_skip = trans_cnt / T_IN_W_T >= (layer_in_h_t - (T_WS - 1));
      if (!col_strip_skip && !row_strip_skip){
        if (!col_skip && !row_skip){

        for (int ii = 0; ii < T_UNROLL; ii++){
          T_data_t0 sum_tmp = sums[ii];
          ap_uint<T_DATA_WIDTH0> utmp_tmp = Reinterpret<ap_uint<T_DATA_WIDTH0> >(sum_tmp);
          utmp[ii] = utmp_tmp;
        }          
        ap_uint<T_DATA_WIDTH0 * T_UNROLL> wide_data = (
#if DEPTH_CONV_LANE == 16
            utmp[15], utmp[14], utmp[13], utmp[12],
            utmp[11], utmp[10], utmp[9], utmp[8],
            utmp[7], utmp[6], utmp[5], utmp[4],
            utmp[3], utmp[2], utmp[1], utmp[0]
#elif DEPTH_CONV_LANE == 8          
            utmp[7], utmp[6], utmp[5], utmp[4],
            utmp[3], utmp[2], utmp[1], utmp[0]
#elif DEPTH_CONV_LANE == 4
            utmp[3], utmp[2], utmp[1], utmp[0]
#elif DEPTH_CONV_LANE == 2           
            utmp[1], utmp[0]
#elif DEPTH_CONV_LANE == 1            
            utmp[0]
#endif
            );
#ifdef DEBUG        
        data_t0 debug_tmpf = Reinterpret<data_t0>(wide_data);
//        cout << col_strip_skip << " " << row_strip_skip << " " << T_IN_W_T << " " << T_WS << endl;
//        cout << "stencil_w3 output: " << debug_tmpf << " trans#: " << trans_cnt << endl;
#endif        
        fifo_out.write(wide_data); 
      
        }
        inner_trans_cnt++;
      }
//      cout << "stencil_w3 trans: " << trans_cnt << endl;
      trans_cnt++;
    }

//  }
    iter++;
    if (iter == iter_bound){
      iter = 0;
      oo++;
      if (oo == oo_bound){
        oo = 0;
      }
    }
  }
}

template <class T_data_t0, class T_data_t1, int T_IN_NUM_T, int T_IN_H_T, int T_IN_W_T, int T_UNROLL, int T_WS, int T_DATA_WIDTH0, int T_DATA_WIDTH1>
void stencil_w1(
  hls::stream<ap_uint<T_DATA_WIDTH0 * T_UNROLL> > &fifo_in,
  T_data_t1                                       weights[T_IN_NUM_T / T_UNROLL][T_UNROLL][K_T][K_T],
  hls::stream<ap_uint<T_DATA_WIDTH0 * T_UNROLL> > &fifo_out,
  uint                                            stride,
  uint                                            layer_in_num_t,
  uint                                            layer_in_h_t
){
#pragma HLS INLINE off
  T_data_t0 line_buf1[T_UNROLL][T_WS];
#pragma HLS ARRAY_PARTITION variable=line_buf1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=line_buf1 dim=2 complete


  ap_uint<T_DATA_WIDTH0> utmp[T_UNROLL];
#pragma HLS ARRAY_PARTITION variable=utmp complete
  T_data_t0 sums[T_UNROLL];
#pragma HLS ARRAY_PARTITION variable=sums complete

  bool col_skip = 0;
  bool row_skip = 0;
  bool col_strip_skip = 0;
  bool row_strip_skip = 0;

  int oo = 0;
  int iter = 0;
  int oo_bound = layer_in_num_t / T_UNROLL;
  int iter_bound = layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1;
  int total_bound = oo_bound * iter_bound;
  uint trans_cnt = 0;
  uint inner_trans_cnt = 0;

//  for (int oo = 0; oo < layer_in_num_t / T_UNROLL; oo++){
//  uint trans_cnt = 0;
//  uint inner_trans_cnt = 0;

//  for (int iter = 0; iter < layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1; iter++){
  for (int total_iter = 0; total_iter < total_bound; total_iter++){
#pragma HLS PIPELINE II=1    
    if (iter == 0){
      trans_cnt = 0;
      inner_trans_cnt = 0;
    }

    ap_uint<T_DATA_WIDTH0 * T_UNROLL> wide_data_in;
    if (iter < layer_in_h_t * T_IN_W_T){
      wide_data_in = fifo_in.read();
    }

    for (int dup = 0; dup < T_UNROLL; dup++){
      for (int i = T_WS - 1; i >= 1; i--){
#pragma HLS UNROLL
        line_buf1[dup][i] = line_buf1[dup][i - 1];
      }
      
      if (iter < layer_in_h_t * T_IN_W_T){        
        ap_uint<T_DATA_WIDTH0> sel_tmp;
#if DEPTH_CONV_LANE == 16
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
          case 4:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 5 - 1, T_DATA_WIDTH0 * 4);
            break;
          case 5:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 6 - 1, T_DATA_WIDTH0 * 5);
            break;
          case 6:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 7 - 1, T_DATA_WIDTH0 * 6);
            break;
          case 7:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 8 - 1, T_DATA_WIDTH0 * 7);
            break;
          case 8:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 9 - 1, T_DATA_WIDTH0 * 8);
            break;
          case 9:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 10 - 1, T_DATA_WIDTH0 * 9);
            break;
          case 10:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 11 - 1, T_DATA_WIDTH0 * 10);
            break;
          case 11:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 12 - 1, T_DATA_WIDTH0 * 11);
            break;
          case 12:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 13 - 1, T_DATA_WIDTH0 * 12);
            break;
          case 13:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 14 - 1, T_DATA_WIDTH0 * 13);
            break;
          case 14:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 15 - 1, T_DATA_WIDTH0 * 14);
            break;
          case 15:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 16 - 1, T_DATA_WIDTH0 * 15);
            break;
        }       
#elif DEPTH_CONV_LANE == 8
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
          case 4:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 5 - 1, T_DATA_WIDTH0 * 4);
            break;
          case 5:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 6 - 1, T_DATA_WIDTH0 * 5);
            break;
          case 6:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 7 - 1, T_DATA_WIDTH0 * 6);
            break;
          case 7:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 8 - 1, T_DATA_WIDTH0 * 7);
            break;
        }
#elif DEPTH_CONV_LANE == 4
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
        }
#elif DEPTH_CONV_LANE == 2
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
        }
#elif DEPTH_CONV_LANE == 1
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
        }
#endif        
        line_buf1[dup][0] = Reinterpret<T_data_t0>(sel_tmp);
      } else {      
        line_buf1[dup][0] = 0.0;
      }

      // mults
      T_data_t0 prod_0_0 = line_buf1[dup][T_WS - 1] * weights[oo][dup][0][0];
      sums[dup]  = prod_0_0;
    }

    if (iter >= (T_WS - 1) * T_IN_W_T + T_WS - 1){      
      col_skip = (inner_trans_cnt % stride != stride - 1);
      row_skip = ((inner_trans_cnt / (T_IN_W_T - (T_WS - 1))) % stride != stride - 1);
      col_strip_skip = trans_cnt % T_IN_W_T >= (T_IN_W_T - (T_WS - 1));
      row_strip_skip = trans_cnt / T_IN_W_T >= (layer_in_h_t - (T_WS - 1));
     
      if (!col_strip_skip && !row_strip_skip){
        if (!col_skip && !row_skip){

        for (int ii = 0; ii < T_UNROLL; ii++){
          T_data_t0 sum_tmp = sums[ii];
          ap_uint<T_DATA_WIDTH0> utmp_tmp = Reinterpret<ap_uint<T_DATA_WIDTH0> >(sum_tmp);
          utmp[ii] = utmp_tmp;
        }          
        ap_uint<T_DATA_WIDTH0 * T_UNROLL> wide_data = (
#if DEPTH_CONV_LANE == 16
            utmp[15], utmp[14], utmp[13], utmp[12],
            utmp[11], utmp[10], utmp[9], utmp[8],
            utmp[7], utmp[6], utmp[5], utmp[4],
            utmp[3], utmp[2], utmp[1], utmp[0]           
#elif DEPTH_CONV_LANE == 8          
            utmp[7], utmp[6], utmp[5], utmp[4],
            utmp[3], utmp[2], utmp[1], utmp[0]
#elif DEPTH_CONV_LANE == 4
            utmp[3], utmp[2], utmp[1], utmp[0]
#elif DEPTH_CONV_LANE == 2           
            utmp[1], utmp[0]
#elif DEPTH_CONV_LANE == 1            
            utmp[0]
#endif
            );
        fifo_out.write(wide_data); 

        }
        inner_trans_cnt++;
      }
      trans_cnt++;
//      if (iter == layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1 - 1){
//        trans_cnt = 0;
//        inner_trans_cnt = 0;
//      }
    }

    iter++;
    if (iter == iter_bound){
      iter = 0;
      oo++;
      if (oo == oo_bound){
        oo = 0;
      }
    }
//  }
  }
}

template <class T_data_t0, int T_IN_H_T, int T_IN_W_T, int T_UNROLL, int T_WS, int T_DATA_WIDTH0>
void maxpool_w2(
  hls::stream<ap_uint<T_DATA_WIDTH0 * T_UNROLL> > &fifo_in,
  hls::stream<ap_uint<T_DATA_WIDTH0 * T_UNROLL> > &fifo_out,
  uint                                            stride,  
  bool                                            max_en,
  uint                                            layer_out_num_t,
  uint                                            layer_in_h_t
){
#pragma HLS INLINE off
  T_data_t0 line_buf1[T_UNROLL][T_IN_W_T];
  T_data_t0 line_buf2[T_UNROLL][T_WS];
#pragma HLS ARRAY_PARTITION variable=line_buf1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=line_buf1 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=line_buf2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=line_buf2 dim=2 complete

  bool col_skip = 0;
  bool row_skip = 0;
  uint trans_cnt = 0;

#ifdef DEBUG
  uint max_pool_cout_cnt = 0;
#endif

  ap_uint<T_DATA_WIDTH0> utmp[T_UNROLL];
#pragma HLS ARRAY_PARTITION variable=utmp complete
  T_data_t0 sums[T_UNROLL];
#pragma HLS ARRAY_PARTITION variable=sums complete

  int oo =0;
  int iter = 0;
  int oo_bound = layer_out_num_t / T_UNROLL;
  int iter_bound = layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1;
  int total_bound = oo_bound * iter_bound;

//  for (int oo = 0; oo < layer_out_num_t / T_UNROLL; oo++){
//  for (int iter = 0; iter < layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1; iter++){
  for (int total_iter = 0; total_iter < total_bound; total_iter++){
#pragma HLS PIPELINE II=1    
    if (iter == 0){
      trans_cnt = 0;
    }

    ap_uint<T_DATA_WIDTH0 * T_UNROLL> wide_data_in;
    if (iter < layer_in_h_t * T_IN_W_T){
      wide_data_in = fifo_in.read();
#ifdef DEBUG
      if (iter == 0){
        T_data_t0 tmpf = Reinterpret<T_data_t0>(wide_data_in);
//        cout << "pool: " << tmpf << endl;
      }
#endif      
    }

    for (int dup = 0; dup < T_UNROLL; dup++){
      T_data_t0 tmp1 = line_buf1[dup][T_IN_W_T - 1];
      for (int i = T_IN_W_T - 1; i >= 1; i--){
#pragma HLS UNROLL
        line_buf1[dup][i] = line_buf1[dup][i - 1];
      }
      for (int i = T_WS - 1; i >= 1; i--){
#pragma HLS UNROLL
        line_buf2[dup][i] = line_buf2[dup][i - 1];
      }
      
      if (iter < layer_in_h_t * T_IN_W_T){        
        ap_uint<T_DATA_WIDTH0> sel_tmp;
#if POOL_LANE == 16
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
          case 4:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 5 - 1, T_DATA_WIDTH0 * 4);
            break;
          case 5:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 6 - 1, T_DATA_WIDTH0 * 5);
            break;
          case 6:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 7 - 1, T_DATA_WIDTH0 * 6);
            break;
          case 7:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 8 - 1, T_DATA_WIDTH0 * 7);
            break;
          case 8:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 9 - 1, T_DATA_WIDTH0 * 8);
            break;
          case 9:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 10 - 1, T_DATA_WIDTH0 * 9);
            break;
          case 10:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 11 - 1, T_DATA_WIDTH0 * 10);
            break;
          case 11:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 12 - 1, T_DATA_WIDTH0 * 11);
            break;
          case 12:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 13 - 1, T_DATA_WIDTH0 * 12);
            break;
          case 13:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 14 - 1, T_DATA_WIDTH0 * 13);
            break;
          case 14:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 15 - 1, T_DATA_WIDTH0 * 14);
            break;
          case 15:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 16 - 1, T_DATA_WIDTH0 * 15);
            break;           
        }
#elif POOL_LANE == 8        
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
          case 4:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 5 - 1, T_DATA_WIDTH0 * 4);
            break;
          case 5:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 6 - 1, T_DATA_WIDTH0 * 5);
            break;
          case 6:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 7 - 1, T_DATA_WIDTH0 * 6);
            break;
          case 7:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 8 - 1, T_DATA_WIDTH0 * 7);
            break;
        }
#elif POOL_LANE == 4
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
          case 2:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 3 - 1, T_DATA_WIDTH0 * 2);
            break;
          case 3:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 4 - 1, T_DATA_WIDTH0 * 3);
            break;
        }
#elif POOL_LANE == 2
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
          case 1:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 2 - 1, T_DATA_WIDTH0 * 1);
            break;
        }
#elif POOL_LANE == 1
        switch(dup){
          case 0:
            sel_tmp = wide_data_in(T_DATA_WIDTH0 * 1 - 1, T_DATA_WIDTH0 * 0);
            break;
        }
#endif     
        line_buf1[dup][0] = Reinterpret<T_data_t0>(sel_tmp);
#ifdef DEBUG
//        cout << "max_pool: " << line_buf1[dup][0] << " idx: " << iter << endl;
#endif        
      } else {      
        line_buf1[dup][0] = 0.0;
      }
      line_buf2[dup][0] = tmp1;

      // maxs 
      T_data_t0 mux_0_0 = max(line_buf2[dup][T_WS - 1], line_buf2[dup][T_WS - 2]);
      T_data_t0 mux_0_1 = max(line_buf1[dup][T_WS - 1], line_buf1[dup][T_WS - 2]);
      T_data_t0 mux_1_0 = max(mux_0_0, mux_0_1);

//      cout << max_en << endl;

      if (max_en == 1)
        sums[dup] = mux_1_0;
      else
        sums[dup] = line_buf1[dup][T_WS - 2];
    }

    if (iter >= (T_WS - 1) * T_IN_W_T + T_WS - 1){      
      col_skip = (trans_cnt % stride != 0);
      row_skip = ((trans_cnt / T_IN_W_T) % stride != 0);
#ifdef DEBUG
//      cout << "trans_cnt: " << trans_cnt << " row_skip: " << row_skip << " col_skip: " << col_skip << " stride: " << stride << endl;
#endif
      if (!col_skip && !row_skip){
        for (int ii = 0; ii < T_UNROLL; ii++){
          T_data_t0 sum_tmp = sums[ii];
//          cout << "max_pool output: " << sum_tmp << " iter: " << iter << endl;
          ap_uint<T_DATA_WIDTH0> utmp_tmp = Reinterpret<ap_uint<T_DATA_WIDTH0> >(sum_tmp);
          utmp[ii] = utmp_tmp;
        }          
        ap_uint<T_DATA_WIDTH0 * T_UNROLL> wide_data = (
#if POOL_LANE == 16
            utmp[15], utmp[14], utmp[13], utmp[12],
            utmp[11], utmp[10], utmp[9], utmp[8],
            utmp[7], utmp[6], utmp[5], utmp[4],
            utmp[3], utmp[2], utmp[1], utmp[0]           
#elif POOL_LANE == 8            
            utmp[7], utmp[6], utmp[5], utmp[4],
            utmp[3], utmp[2], utmp[1], utmp[0]
#elif POOL_LANE == 4
            utmp[3], utmp[2], utmp[1], utmp[0]
#elif POOL_LANE == 2
            utmp[1], utmp[0]
#elif POOL_LANE == 1
            utmp[0]
#endif            
            );
#ifdef DEBUG
//        if (trans_cnt == 0)
//          cout << "max_pool: " << sums[0] << endl;
#endif
        fifo_out.write(wide_data); 
#ifdef DEBUG
        max_pool_cout_cnt++;
#endif
      }
      trans_cnt++;
//      if (iter == layer_in_h_t * T_IN_W_T + (T_WS - 1) * T_IN_W_T + T_WS - 1 - 1){
//        trans_cnt = 0;
//      }
    }

    iter++;
    if (iter == iter_bound){
      iter = 0;
      oo++;
      if (oo == oo_bound){
        oo = 0;
      }
    }
//  }
  }
#ifdef DEBUG
//  cout << "max_pool: " << max_pool_cout_cnt << endl;
#endif 
}

#endif
