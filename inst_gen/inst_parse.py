from math import ceil
import json
import argparse

# TODO: LAYER1_K, STAGE2L_K, STAGE2R_K, STAGE2L_OFFSET, STAGE2R_OFFSET

# Array layout:
# cin[IN_NUM_HW / IN_NUM_T][H_HW][W_HW][IN_NUM_T]
# w[OUT_NUM_HW][IN_NUM_HW / IN_NUM_T][K][K][IN_NUM_T]
# bias[OUT_NUM_HW]
# cout[OUT_NUM_HW / OUT_NUM_T][H_HW][W_HW][OUT_NUM_T]

# Memory Layout:
# ----------------
# input figure
# ----------------
# *** REGION 0 ***
# conv2d_0
# conv2d_1
# conv2d_2
# conv2d_3
# conv2d_4
# conv2d_5
# conv2d_6
# ----------------
# *** REGION 1 ***
# conv2d_8
# conv2d_9
# conv2d_10
# ----------------
# *** REGION 2 ***
# MConv_Stage1_L1_[1-4]
# MConv_Stage1_L2_[1-4]
# ----------------
# *** REGION 3 ***
# MConv_Stage[2-6]_L1_[1-4]
# MConv_Stage[2-6]_L1_[1-4]
# ----------------
# *** REGION 4 ***
# MConv_Stage1_L1_5 / MConv_Stage[3,5]_L1_5
# MConv_Stage1_L2_5 / MConv_Stage[3,5]_L2_5
# conv2d_3_pool
# conv2d_7
# conv2d_11
# ----------------
# *** REGION 5 ***
# MConv_Stage[2,4,6]_L1_5
# MConv_Stage[2,4,6]_L1_5
# ----------------

# Instruction Layout:
# inst0: in_num_hw | out_num_hw | in_h_hw | in_w_hw | out_h_hw | out_w_hw
# inst1: in_num | out_num | in_h | in_w | out_h | out_w
# inst2: cin_offset | weight_offset | bias_offset | cout_offset | filter_s1 | filter_s2 | stride
# inst3: layer_en: depth_conv_en, conv_en, relu_en, pool_en, up_sample_en, bias_en | in_num_t | out_num_t | layer_batch | in_h_t | in_w_t

def run(f_tile, f_model, f_model_config, f_input_config):

  macros = open("./params.h", "w")

  model = open(f_model, "r")
  with open(f_tile, "r") as f:
    tile = json.loads(f.read())
  with open(f_model_config, "r") as f:
    model_config = json.loads(f.read())
  with open(f_input_config, "r") as f:
    input_config = json.loads(f.read())

  # Please change the paramters below before running this script
  """
  Tiling Size
  """
  IN_NUM_T = tile["IN_NUM_T"]
  OUT_NUM_T = tile["OUT_NUM_T"]
  IN_H_T = tile["IN_H_T"]
  IN_W_T = tile["IN_W_T"]
  OUT_H_T = tile["OUT_H_T"]
  OUT_W_T = tile["OUT_W_T"]
  K_T = tile["K_T"]
  SA_ROWS = tile["SA_ROWS"]
  SA_COLS = tile["SA_COLS"]
  SA_SIMD = tile["SA_SIMD"]

  """
  Model Params
  """
  # openpose_thin model
  VGG_LAYERS = model_config["VGG_LAYERS"]
  STAGE1_LAYERS = model_config["STAGE1_LAYERS"]
  STAGE1_ITER = model_config["STAGE1_ITER"]
  STAGE2_LAYERS = model_config["STAGE2_LAYERS"]
  STAGE2_ITER = model_config["STAGE2_ITER"]

  # input info
  network_in_num = input_config["IN_NUM"]
  network_in_h = input_config["IN_H"]
  network_in_w = input_config["IN_W"]

  # Please do not change the code below
  # write out macros
  macros.write("#define IN_NUM_T " + str(IN_NUM_T) + '\n')
  macros.write("#define OUT_NUM_T " + str(OUT_NUM_T) + '\n')
  macros.write("#define IN_H_T " + str(IN_H_T) + '\n')
  macros.write("#define IN_W_T " + str(IN_W_T) + '\n')
  macros.write("#define OUT_H_T " + str(OUT_H_T) + '\n')
  macros.write("#define OUT_W_T " + str(OUT_W_T) + '\n')
  macros.write("#define K_T " + str(K_T) + '\n')

  LAYER_NUM = VGG_LAYERS + STAGE1_LAYERS*2 + STAGE2_LAYERS * 2 * STAGE2_ITER
  macros.write("#define LAYER_NUM " + str(LAYER_NUM) + '\n')
  macros.write("#define VGG_LAYERS " + str(VGG_LAYERS) + '\n')
  macros.write("#define STAGE1_LAYERS " + str(STAGE1_LAYERS) + '\n')
  macros.write("#define STAGE1_ITER " + str(STAGE1_ITER) + '\n')
  macros.write("#define STAGE2_LAYERS " + str(STAGE2_LAYERS) + '\n')
  macros.write("#define STAGE2_ITER " + str(STAGE2_ITER) + '\n')

  #model = open("./small.model", "r")
  insts = open("./openpose.insts", "w")
  weight_load = open("./weight_offset.dat", "w")
  bias_load = open("./bias_offset.dat", "w")

  lines = []
  for i in model.readlines():
    lines.append(i)

  line_num = len(lines)

  vgg_layer_cnt = 0
  stage1_layer_cnt = 0
  stage1_iter_cnt = 0
  stage2_layer_cnt = 0
  stage2_iter_cnt = 0
  stage1_channel_cnt = 0
  stage2_channel_cnt = 0

  """
  Offset Calculation
  cin_offset <- cin_offset + IN_NUM_HW(prev_layer)*IN_H_HW(prev_layer)*IN_W_HW(prev_layer)
  weight_offset <- weight_offset + OUT_NUM_HW(prev_layer)*IN_NUM_HW(prev_layer)*FILTER_S(prev_layer)*FILTER_S(prev_layer)
  bias_offset <- bias_offset + OUT_NUM_HW(prev_layer)
  cout_offset <- cout_offset + OUT_NUM_HW(prev_layer)*OUT_H_HW(prev_layer)*OUT_W_HW(prev_layer)
  """
  cin_offset = 0
  cout_offset = 0
  weight_offset = 0
  bias_offset = 0

  region0_offset = 0
  region1_offset = 0
  region2_offset = 0
  region3_offset = 0
  region4_offset = 0

  current_model = "VGG" #{VGG, STAGE1, STAGE2}
  stage1_line_id = 0
  stage2_line_id = 0

  in_num_hw = 0
  out_num_hw = 0
  in_h_hw = 0
  in_w_hw = 0
  out_h_hw = 0
  out_w_hw = 0
  filter_s = 0

  in_num = network_in_num
  out_num = network_in_num
  in_h = network_in_h
  in_w = network_in_w
  out_h = network_in_h
  out_w = network_in_w

  in_out_offset = 0

  # Pass 1: To learn about the filter size, in_out_offset, in_h_t, in_w_t
  # We need a separate pass to learn about the filter size
  filter_list = []
  in_h_t_list = []
  in_w_t_list = []
  for line_id in range(1, len(lines)):
    line = lines[line_id].strip('\n')
    content = line.split(",")
    if len(content) > 1:
      filter_s = int(content[3])
      filter_list.append(filter_s)
      in_h_t = int(content[9])
      in_w_t = int(content[10])
      in_h_t_list.append(in_h_t)
      in_w_t_list.append(in_w_t)

  #print filter_list
  # Get the in_out_offset
  line_id = 1
  while line_id < len(lines):
    line = lines[line_id].strip('\n')
    content = line.split(",")
    if current_model == "VGG":
      if content[1] == "convb" or content[1] == "separable_conv":
        if vgg_layer_cnt == 1:
  	      in_out_offset = in_num_hw * in_h_hw * in_w_hw

        if vgg_layer_cnt == 1:
          macros.write("#define IN_OUT_OFFSET " + str(int(in_out_offset)) + '\n')

        relu_en = 0
        pool_en = 0
        if len(content) > 1 and (content[5] == "1"):
          relu_en = 1
        if content[1] == "max_pool":
          pool_en = 1
        layer_name = content[0]
        layer_type = content[1]

        in_num = out_num
        out_num = int(content[2])
        filter_s = int(content[3])
        stride = int(content[4])
        in_h = out_h
        in_w = out_w
        in_num_t = int(content[7])
        out_num_t = int(content[8])
        in_h_t = int(content[9])
        in_w_t = int(content[10])

        if stride == 2:
          out_h = int(ceil(float(in_h) / 2))
          out_w = int(ceil(float(in_w) / 2))
        else:
          out_h = in_h
          out_w = in_w

        cur_filter_s = filter_s
        nxt_filter_s = 0
        if layer_name == "Conv2d_7" or layer_name == "Conv2d_11" or layer_name == "Conv2d_3_pool":
          nxt_filter_s = max(filter_list[VGG_LAYERS], filter_list[VGG_LAYERS + STAGE1_LAYERS * 2])
          out_h_t = in_h_t_list[VGG_LAYERS]
          out_w_t = in_w_t_list[VGG_LAYERS]
        else:
          nxt_filter_s = filter_list[vgg_layer_cnt + 1]
          out_h_t = in_h_t_list[vgg_layer_cnt + 1]
          out_w_t = in_w_t_list[vgg_layer_cnt + 1]

        in_num_hw = int(ceil(float(in_num) / in_num_t) * in_num_t)
        out_num_hw = int(ceil(float(out_num) / out_num_t) * out_num_t)
        in_h_hw = int(ceil(float(in_h) / in_h_t) * in_h_t + (cur_filter_s - 1))
        in_w_hw = int(ceil(float(in_w) / in_w_t) * in_w_t + (cur_filter_s - 1))
        out_h_hw = int(ceil(float(out_h) / out_h_t) * out_h_t + (nxt_filter_s - 1))
        out_w_hw = int(ceil(float(out_w) / out_w_t) * out_w_t + (nxt_filter_s - 1))

#        print(in_num_hw, in_h_hw, in_w_hw)
#        print(vgg_layer_cnt)

        vgg_layer_cnt = vgg_layer_cnt + 1
  	    # store the start line number of stage 1
        if vgg_layer_cnt == 2:
          break
    line_id = line_id + 1

#  print(in_out_offset)

  # reinitialize all parameters
  vgg_layer_cnt = 0
  vgg_layer_cnt = 0
  stage1_layer_cnt = 0
  stage1_iter_cnt = 0
  stage2_layer_cnt = 0
  stage2_iter_cnt = 0
  stage1_channel_cnt = 0
  stage2_channel_cnt = 0

  in_num_hw = 0
  out_num_hw = 0
  in_h_hw = 0
  in_w_hw = 0
  out_h_hw = 0
  out_w_hw = 0
  filter_s = 0

  in_num = network_in_num
  out_num = network_in_num
  in_h = network_in_h
  in_w = network_in_w
  out_h = network_in_h
  out_w = network_in_w

  # Pass 2: To learn about layer output size
  layer_cin_size = {}
  layer_cin_size_hw = {}
  layer_cout_size = {}
  layer_cout_size_hw = {}
  layer_cout_size_hw_concat = {}
  layer_weight_size = {}
  layer_weight_size_hw = {}
  layer_bias_size = {}
  layer_bias_size_hw = {}

  Conv2d_3_out_num = 0
  Conv2d_3_out_h = 0
  Conv2d_3_out_w = 0
  Conv2d_3_pool_out_num = 0
  Conv2d_3_pool_out_h = 0
  Conv2d_3_pool_out_w = 0
  Conv2d_7_out_num = 0
  Conv2d_7_out_h = 0
  Conv2d_7_out_w = 0
  Conv2d_11_out_num = 0
  Conv2d_11_out_h = 0
  Conv2d_11_out_w = 0
  MConv_Stage1_L1_5_out_num = 0
  MConv_Stage1_L1_5_out_h = 0
  MConv_Stage1_L1_5_out_w = 0
  MConv_Stage1_L2_5_out_num = 0
  MConv_Stage1_L2_5_out_h = 0
  MConv_Stage1_L2_5_out_w = 0
  MConv_Stage2_L1_5_out_num = 0
  MConv_Stage2_L1_5_out_h = 0
  MConv_Stage2_L1_5_out_w = 0
  MConv_Stage2_L2_5_out_num = 0
  MConv_Stage2_L2_5_out_h = 0
  MConv_Stage2_L2_5_out_w = 0

  max_layer_batch = 1
  layer_configs = {}

  #for line_id in range(0,len(lines)):
  line_id = 1
  while line_id < len(lines):
    line = lines[line_id].strip('\n')
    content = line.split(",")
    if current_model == "VGG":
      relu_en = 0
      pool_en = 0
      bias_en = 0
      if len(content) > 1 and content[6] == "1":
        bias_en = 1
      if len(content) > 1 and content[5] == "1":
        relu_en = 1
#      if len(content) > 1 and content[4] == "2":
#        pool_en = 1
      if len(content) > 1 and content[1] == "max_pool":
        pool_en = 1
      layer_name = content[0]
      layer_type = content[1]

      #print(layer_name)

      in_num = out_num
      in_h = out_h
      in_w = out_w
      out_num = int(content[2])
      filter_s = int(content[3])
      stride = int(content[4])
      in_num_t = int(content[7])
      out_num_t = int(content[8])
      in_h_t = int(content[9])
      in_w_t = int(content[10])

      if layer_name == "Conv2d_3_pool":
        in_num = Conv2d_3_out_num
        in_h = Conv2d_3_out_h
        in_w = Conv2d_3_out_w
        filter_s = Conv2d_3_nxt_filter_s

      if stride == 2:
        out_h = int(ceil(float(in_h) / 2))
        out_w = int(ceil(float(in_w) / 2))
#        out_h = (in_h + 1) / 2
#        out_w = (in_w + 1) / 2
      else:
        out_h = in_h
        out_w = in_w

      cur_filter_s = filter_s
      nxt_filter_s = 0
      #if vgg_layer_cnt == VGG_LAYERS - 2:
      if layer_name == "Conv2d_7" or layer_name == "Conv2d_11" or layer_name == "Conv2d_3_pool":
        nxt_filter_s = max(filter_list[VGG_LAYERS], filter_list[VGG_LAYERS + STAGE1_LAYERS * 2])
        out_h_t = in_h_t_list[VGG_LAYERS]
        out_w_t = in_w_t_list[VGG_LAYERS]
      else:
        nxt_filter_s = filter_list[vgg_layer_cnt + 1]
        out_h_t = in_h_t_list[vgg_layer_cnt + 1]
        out_w_t = in_w_t_list[vgg_layer_cnt + 1]

      in_num_hw = int(ceil(float(in_num) / in_num_t) * in_num_t)
      out_num_hw = int(ceil(float(out_num) / out_num_t) * out_num_t)
      in_h_hw = int(ceil(float(in_h) / in_h_t) * in_h_t + (cur_filter_s - 1))
      in_w_hw = int(ceil(float(in_w) / in_w_t) * in_w_t + (cur_filter_s - 1))
      out_h_hw = int(ceil(float(out_h) / out_h_t) * out_h_t + (nxt_filter_s - 1))
      out_w_hw = int(ceil(float(out_w) / out_w_t) * out_w_t + (nxt_filter_s - 1))

      task_num1 = int(ceil(float(in_num) / in_num_t) * ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
      task_num2 = int(ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
      if layer_type == "separable_conv":
        local_accum_num = int(in_num_t / SA_SIMD * 1 * 1)
      else:
        local_accum_num = int(in_num_t / SA_SIMD * filter_s * filter_s)
      local_reg_num = int((in_h_t / stride) * (in_w_t / SA_COLS / stride) * (out_num_t / SA_ROWS))
      row_il_factor = int(out_num_t / SA_ROWS)
      col_il_factor = int(in_w_t / SA_COLS / stride)

      if layer_name == "Conv2d_3":
        Conv2d_3_out_num = out_num
        Conv2d_3_out_num_hw = out_num_hw
        Conv2d_3_out_h = out_h
        Conv2d_3_out_w = out_w
        Conv2d_3_nxt_filter_s = nxt_filter_s

      if layer_name == "Conv2d_7":
        Conv2d_7_out_num = out_num
        Conv2d_7_out_num_hw = out_num_hw
        Conv2d_7_out_h = out_h
        Conv2d_7_out_w = out_w
        Conv2d_7_nxt_filter_s = nxt_filter_s

      if layer_name == "Conv2d_11":
        Conv2d_11_out_num = out_num
        Conv2d_11_out_num_hw = out_num_hw
        Conv2d_11_out_h = out_h
        Conv2d_11_out_w = out_w
        Conv2d_11_nxt_filter_s = nxt_filter_s

      if layer_name == "Conv2d_3_pool":
        Conv2d_3_pool_out_num = out_num
        Conv2d_3_pool_out_num_hw = out_num_hw
        Conv2d_3_pool_out_h = out_h
        Conv2d_3_pool_out_w = out_w
        Conv2d_3_pool_nxt_filter_s = nxt_filter_s

      # calculate cin, cout, weight, bias size
      cin_size = in_num * in_h * in_w
      cin_size_hw = in_num_hw * in_h_hw * in_w_hw

      cout_size = out_num * out_h * out_w
      cout_size_hw = out_num_hw * out_h_hw * out_w_hw
      cout_size_hw_concat = out_num * out_h_hw * out_w_hw

      if layer_type == "separable_conv":
        weight_size = in_num * filter_s * filter_s + in_num * out_num * 1 * 1
        weight_size_hw = in_num_hw * filter_s * filter_s + in_num_hw *out_num_hw * 1 * 1
        bias_size = out_num
        bias_size_hw = out_num_hw
      elif layer_type == "convb":
        weight_size = in_num * out_num * filter_s * filter_s
        weight_size_hw = in_num_hw * out_num_hw * filter_s * filter_s
        bias_size = out_num
        bias_size_hw = out_num_hw
      else: # maxpool
        weight_size = 0
        weight_size_hw = 0
        bias_size = 0
        bias_size_hw = 0

      weight_load_size = weight_size_hw
      bias_load_size = bias_size_hw
      weight_load.write(str(weight_offset) + " " + str(weight_load_size) + '\n')
      bias_load.write(str(bias_offset) + " " + str(bias_load_size) + '\n')

      weight_offset = weight_offset + weight_load_size
      bias_offset = bias_offset + bias_load_size

      if vgg_layer_cnt == 0:
        macros.write("#define LAYER1_IN_NUM " + str(in_num) + '\n')
        macros.write("#define LAYER1_OUT_NUM " + str(out_num) + '\n')
        macros.write("#define LAYER1_IN_NUM_T " + str(in_num_t) + '\n')
        macros.write("#define LAYER1_OUT_NUM_T " + str(out_num_t) + '\n')
        macros.write("#define LAYER1_IN_H " + str(in_h) + '\n')
        macros.write("#define LAYER1_IN_W " + str(in_w) + '\n')
        macros.write("#define LAYER1_OUT_H " + str(out_h) + '\n')
        macros.write("#define LAYER1_OUT_W " + str(out_w) + '\n')
        macros.write("#define LAYER1_IN_NUM_HW " + str(int(in_num_hw)) + '\n')
        macros.write("#define LAYER1_OUT_NUM_HW " + str(int(out_num_hw)) + '\n')
        macros.write("#define LAYER1_IN_H_HW " + str(int(in_h_hw)) + '\n')
        macros.write("#define LAYER1_IN_W_HW " + str(int(in_w_hw)) + '\n')
        macros.write("#define LAYER1_OUT_H_HW " + str(int(out_h_hw)) + '\n')
        macros.write("#define LAYER1_OUT_W_HW " + str(int(out_w_hw)) + '\n')
#        macros.write("#define LAYER1_K " + str(filter_s) + '\n')
        macros.write("#define LAYER1_K " + str(nxt_filter_s) + '\n')
        macros.write("#define LAYER1_POOL " + str(pool_en) + '\n')

      layer_cin_size[layer_name] = cin_size
      layer_cin_size_hw[layer_name] = cin_size_hw

      layer_cout_size[layer_name] = cout_size
      layer_cout_size_hw[layer_name] = cout_size_hw
      layer_cout_size_hw_concat[layer_name] = cout_size_hw_concat

      layer_weight_size[layer_name] = weight_size
      layer_weight_size_hw[layer_name] = weight_size_hw

      layer_bias_size[layer_name] = bias_size
      layer_bias_size_hw[layer_name] = bias_size_hw

      layer_config = {}
      layer_config['IN_NUM'] = in_num
      layer_config['OUT_NUM'] = out_num
      layer_config['IN_H'] = in_h
      layer_config['IN_W'] = in_w
      layer_config['OUT_H'] = out_h
      layer_config['OUT_W'] = out_w
      layer_config['IN_NUM_HW'] = in_num_hw
      layer_config['OUT_NUM_HW'] = out_num_hw
      layer_config['IN_H_HW'] = in_h_hw
      layer_config['IN_W_HW'] = in_w_hw
      layer_config['OUT_H_HW'] = out_h_hw
      layer_config['OUT_W_HW'] = out_w_hw
      if layer_type == "separable_conv":
        layer_config['FILTER_S1'] = filter_s
        layer_config['FILTER_S2'] = 1
      elif layer_type == "convb":
        layer_config['FILTER_S1'] = 1
        layer_config['FILTER_S2'] = filter_s
      elif layer_type == "max_pool":
        layer_config['FILTER_S1'] = 1
        layer_config['FILTER_S2'] = 1
      layer_config['STRIDE'] = stride
      layer_config['IN_NUM_T'] = in_num_t
      layer_config['OUT_NUM_T'] = out_num_t
      layer_config['IN_H_T'] = in_h_t
      layer_config['IN_W_T'] = in_w_t
      layer_config['NXT_LAYER_BATCH'] = 1
      layer_config['TASK_NUM1'] = task_num1
      layer_config['TASK_NUM2'] = task_num2
      layer_config['LOCAL_ACCUM_NUM'] = local_accum_num
      layer_config['LOCAL_REG_NUM'] = local_reg_num
      layer_config['ROW_IL_FACTOR'] = row_il_factor
      layer_config['COL_IL_FACTOR'] = col_il_factor

      if layer_type == "separable_conv":
        depth_conv_en = 1
        conv_en = 1
      elif layer_type == "convb":
        depth_conv_en = 0
        conv_en = 1
      elif layer_type == "max_pool":
        depth_conv_en = 0
        conv_en = 0

      up_sample_en = 0
      inter_load_en = 0
      inter_write_en = 0

      layer_en = depth_conv_en + (conv_en << 1) + (relu_en << 2) + (pool_en << 3) + (up_sample_en << 4) + (bias_en << 5) + (inter_load_en << 6) + (inter_write_en << 7)

#      if vgg_layer_cnt == 0:
#        print(depth_conv_en)
#        print(conv_en << 1)
#        print(relu_en << 2)
#        print(pool_en << 3)
#        print(up_sample_en << 4)
#        print(ld_sel << 5)
#        print(wr_sel << 6)
#        print(depth_conv_en, conv_en, relu_en, pool_en, up_sample_en, ld_sel, wr_sel, layer_en)

      layer_config['LAYER_EN'] = layer_en
      layer_configs[layer_name] = layer_config

      vgg_layer_cnt = vgg_layer_cnt + 1
  	  # store the start line number of stage 1
      if vgg_layer_cnt == VGG_LAYERS:
        current_model = "STAGE1"

    elif current_model == "STAGE1":
      relu_en = 0
      pool_en = 0
      bias_en = 0
      if len(content) > 1 and content[6] == "1":
        bias_en = 1
      if len(content) > 1 and content[5] == "1":
        relu_en = 1
#      if len(content) > 1 and content[4] == "2":
#        pool_en = 1
      if len(content) > 1 and content[1] == "max_pool":
        pool_en = 1
      layer_name = content[0]
      layer_type = content[1]

      #print(layer_name)

      in_num = out_num
      in_h = out_h
      in_w = out_w
      out_num = int(content[2])
      filter_s = int(content[3])
      stride = int(content[4])
      in_num_t = int(content[7])
      out_num_t = int(content[8])
      in_h_t = int(content[9])
      in_w_t = int(content[10])

      if stage1_layer_cnt == 0:
        in_num = Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw + Conv2d_11_out_num_hw
        in_h = Conv2d_3_pool_out_h
        in_w = Conv2d_3_pool_out_w

      if stride == 2:
        out_h = int(ceil(float(in_h) / 2))
        out_w = int(ceil(float(in_w) / 2))
#        out_h = (in_h + 1) / 2
#        out_w = (in_w + 1) / 2
      else:
        out_h = in_h
        out_w = in_w

      cur_filter_s = filter_s
      nxt_filter_s = 0
      if layer_name == "MConv_Stage1_L1_5" or layer_name == "MConv_Stage1_L2_5":
        nxt_filter_s = filter_list[VGG_LAYERS + STAGE1_LAYERS * 2]
        out_h_t = in_h_t_list[VGG_LAYERS + STAGE1_LAYERS * 2]
        out_w_t = in_w_t_list[VGG_LAYERS + STAGE1_LAYERS * 2]
      else:
        nxt_filter_s = filter_list[VGG_LAYERS + stage1_layer_cnt + stage1_channel_cnt * STAGE1_LAYERS + 1]
        out_h_t = in_h_t_list[VGG_LAYERS + stage1_layer_cnt + stage1_channel_cnt * STAGE1_LAYERS + 1]
        out_w_t = in_w_t_list[VGG_LAYERS + stage1_layer_cnt + stage1_channel_cnt * STAGE1_LAYERS + 1]

      in_num_hw = int(ceil(float(in_num) / in_num_t) * in_num_t)
      out_num_hw = int(ceil(float(out_num) / out_num_t) * out_num_t)
      in_h_hw = int(ceil(float(in_h) / in_h_t) * in_h_t + (cur_filter_s - 1))
      in_w_hw = int(ceil(float(in_w) / in_w_t) * in_w_t + (cur_filter_s - 1))
      out_h_hw = int(ceil(float(out_h) / out_h_t) * out_h_t + (nxt_filter_s - 1))
      out_w_hw = int(ceil(float(out_w) / out_w_t) * out_w_t + (nxt_filter_s - 1))

      task_num1 = int(ceil(float(in_num) / in_num_t) * ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
      task_num2 = int(ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
      if layer_type == "separable_conv":
        local_accum_num = int(in_num_t / SA_SIMD * 1 * 1)
      else:
        local_accum_num = int(in_num_t / SA_SIMD * filter_s * filter_s)
#      local_accum_num = int(in_num_t / SA_SIMD * filter_s * filter_s)
      local_reg_num = int((in_h_t / stride) * (in_w_t / SA_COLS / stride) * (out_num_t / SA_ROWS))
      row_il_factor = int(out_num_t / SA_ROWS)
      col_il_factor = int(in_w_t / SA_COLS / stride)

      if layer_name == "MConv_Stage1_L1_5":
        MConv_Stage1_L1_5_out_num = out_num
        MConv_Stage1_L1_5_out_num_hw = out_num_hw
        MConv_Stage1_L1_5_out_h = out_h
        MConv_Stage1_L1_5_out_w = out_w

      if layer_name == "MConv_Stage1_L2_5":
        MConv_Stage1_L2_5_out_num = out_num
        MConv_Stage1_L2_5_out_num_hw = out_num_hw
        MConv_Stage1_L2_5_out_h = out_h
        MConv_Stage1_L2_5_out_w = out_w

      # calculate cin, cout, weight, bias size
      cin_size = in_num * in_h * in_w
      cin_size_hw = in_num_hw * in_h_hw * in_w_hw

      cout_size = out_num * out_h * out_w
      cout_size_hw = out_num_hw * out_h_hw * out_w_hw
      cout_size_hw_concat = out_num * out_h_hw * out_w_hw

      if layer_type == "separable_conv":
        weight_size = in_num * filter_s * filter_s + in_num * out_num * 1 * 1
        weight_size_hw = in_num_hw * filter_s * filter_s + in_num_hw *out_num_hw * 1 * 1
        bias_size = out_num
        bias_size_hw = out_num_hw
      elif layer_type == "convb":
        weight_size = in_num * out_num * filter_s * filter_s
        weight_size_hw = in_num_hw * out_num_hw * filter_s * filter_s
        bias_size = out_num
        bias_size_hw = out_num_hw
      else: # maxpool
        weight_size = 0
        weight_size_hw = 0
        bias_size = 0
        bias_size_hw = 0

      weight_load_size = weight_size_hw
      bias_load_size = bias_size_hw
      weight_load.write(str(weight_offset) + " " + str(weight_load_size) + '\n')
      bias_load.write(str(bias_offset) + " " + str(bias_load_size) + '\n')

      weight_offset = weight_offset + weight_load_size
      bias_offset = bias_offset + bias_load_size

      layer_cin_size[layer_name] = cin_size
      layer_cin_size_hw[layer_name] = cin_size_hw

      layer_cout_size[layer_name] = cout_size
      layer_cout_size_hw[layer_name] = cout_size_hw
      layer_cout_size_hw_concat[layer_name] = cout_size_hw_concat

      layer_weight_size[layer_name] = weight_size
      layer_weight_size_hw[layer_name] = weight_size_hw

      layer_bias_size[layer_name] = bias_size
      layer_bias_size_hw[layer_name] = bias_size_hw

      layer_config = {}
      layer_config['IN_NUM'] = in_num
      layer_config['OUT_NUM'] = out_num
      layer_config['IN_H'] = in_h
      layer_config['IN_W'] = in_w
      layer_config['OUT_H'] = out_h
      layer_config['OUT_W'] = out_w
      layer_config['IN_NUM_HW'] = in_num_hw
      layer_config['OUT_NUM_HW'] = out_num_hw
      layer_config['IN_H_HW'] = in_h_hw
      layer_config['IN_W_HW'] = in_w_hw
      layer_config['OUT_H_HW'] = out_h_hw
      layer_config['OUT_W_HW'] = out_w_hw
      if layer_type == "separable_conv":
        layer_config['FILTER_S1'] = filter_s
        layer_config['FILTER_S2'] = 1
      elif layer_type == "convb":
        layer_config['FILTER_S1'] = 1
        layer_config['FILTER_S2'] = filter_s
      elif layer_type == "max_pool":
        layer_config['FILTER_S1'] = 1
        layer_config['FILTER_S2'] = 1
      layer_config['STRIDE'] = stride
      layer_config['IN_NUM_T'] = in_num_t
      layer_config['OUT_NUM_T'] = out_num_t
      layer_config['IN_H_T'] = in_h_t
      layer_config['IN_W_T'] = in_w_t
      layer_config['NXT_LAYER_BATCH'] = 1
      layer_config['TASK_NUM1'] = task_num1
      layer_config['TASK_NUM2'] = task_num2
      layer_config['LOCAL_ACCUM_NUM'] = local_accum_num
      layer_config['LOCAL_REG_NUM'] = local_reg_num
      layer_config['ROW_IL_FACTOR'] = row_il_factor
      layer_config['COL_IL_FACTOR'] = col_il_factor

      if layer_type == "separable_conv":
        depth_conv_en = 1
        conv_en = 1
      elif layer_type == "convb":
        depth_conv_en = 0
        conv_en = 1
      elif layer_type == "max_pool":
        depth_conv_en = 0
        conv_en = 0

      up_sample_en = 0
#      ld_sel = 0
#      wr_sel = 0
      inter_load_en = 0
      inter_write_en = 0

#      if stage1_layer_cnt == 0:
#        inter_write_en = 1
#        inter_load_en = 0
#      elif stage1_layer_cnt == 2:
#        inter_write_en = 0
#        inter_load_en = 1
#      elif stage1_layer_cnt == 1:
#        inter_write_en = 1
#        inter_load_en = 1
#      else:
#        inter_write_en = 0
#        inter_load_en = 0

      if stage1_layer_cnt < STAGE1_LAYERS - 1:
        if stage1_channel_cnt == 1:
          layer_config['NXT_LAYER_BATCH'] = 2
          max_layer_batch = max(max_layer_batch, 2)

      layer_en = depth_conv_en + (conv_en << 1) + (relu_en << 2) + (pool_en << 3) + (up_sample_en << 4) + (bias_en << 5) + (inter_load_en << 6) + (inter_write_en << 7)

      layer_config['LAYER_EN'] = layer_en
      layer_configs[layer_name] = layer_config

      stage1_layer_cnt = stage1_layer_cnt + 1
      if stage1_layer_cnt == STAGE1_LAYERS:
        stage1_layer_cnt = 0
        stage1_channel_cnt = stage1_channel_cnt + 1
        if stage1_channel_cnt == 2:
          stage1_channel_cnt = 0
          stage1_iter_cnt = stage1_iter_cnt + 1
          if stage1_iter_cnt == STAGE1_ITER:
            stage1_iter_cnt = 0
            stage2_line_id = line_id + 1
            current_model = "STAGE2"

## Change the execution order of two branches
#      stage1_channel_cnt = stage1_channel_cnt + 1
#      if stage1_channel_cnt == 2:
#        stage1_channel_cnt = 0
#        stage1_layer_cnt = stage1_layer_cnt + 1
#        if stage1_layer_cnt == STAGE1_LAYERS:
#          stage1_layer_cnt = 0
#          stage1_iter_cnt = stage1_iter_cnt + 1
#          if stage1_iter_cnt == STAGE1_ITER:
#            stage1_iter_cnt = 0
#            stage2_line_id = line_id + 1
#            current_model == "STAGE2"
    elif current_model == "STAGE2":
      relu_en = 0
      pool_en = 0
      bias_en = 0
      if len(content) > 1 and content[6] == "1":
        bias_en = 1
      if len(content) > 1 and content[5] == "1":
        relu_en = 1
#      if len(content) > 1 and content[4] == "2":
#        pool_en = 1
      if len(content) > 1 and content[1] == "max_pool":
        pool_en = 1
      layer_name = content[0]
      layer_type = content[1]

      #print(layer_name)

      in_num = out_num
      in_h = out_h
      in_w = out_w
      out_num = int(content[2])
      filter_s = int(content[3])
      stride = int(content[4])
      in_num_t = int(content[7])
      out_num_t = int(content[8])
      in_h_t = int(content[9])
      in_w_t = int(content[10])

      if stage2_layer_cnt == 0:
        in_num = MConv_Stage1_L1_5_out_num_hw + MConv_Stage1_L2_5_out_num_hw + Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw + Conv2d_11_out_num_hw
        in_h = Conv2d_3_pool_out_h
        in_w = Conv2d_3_pool_out_w

      if stride == 2:
        out_h = int(ceil(float(in_h) / 2))
        out_w = int(ceil(float(in_w) / 2))
#        out_h = (in_h + 1) / 2
#        out_w = (in_w + 1) / 2
      else:
        out_h = in_h
        out_w = in_w

      cur_filter_s = filter_s
      nxt_filter_s = 0
      if layer_name == "MConv_Stage2_L1_5" or layer_name == "MConv_Stage2_L2_5":
        nxt_filter_s = filter_list[VGG_LAYERS + STAGE1_LAYERS * 2]
        out_h_t = in_h_t_list[VGG_LAYERS + STAGE1_LAYERS * 2]
        out_w_t = in_w_t_list[VGG_LAYERS + STAGE1_LAYERS * 2]
      else:
        nxt_filter_s = filter_list[VGG_LAYERS + STAGE1_LAYERS * 2 + stage2_layer_cnt + stage2_channel_cnt * STAGE2_LAYERS + 1]
        out_h_t = in_h_t_list[VGG_LAYERS + STAGE1_LAYERS * 2 + stage2_layer_cnt + stage2_channel_cnt * STAGE2_LAYERS + 1]
        out_w_t = in_w_t_list[VGG_LAYERS + STAGE1_LAYERS * 2 + stage2_layer_cnt + stage2_channel_cnt * STAGE2_LAYERS + 1]

      in_num_hw = int(ceil(float(in_num) / in_num_t) * in_num_t)
      out_num_hw = int(ceil(float(out_num) / out_num_t) * out_num_t)
      in_h_hw = int(ceil(float(in_h) / in_h_t) * in_h_t + (cur_filter_s - 1))
      in_w_hw = int(ceil(float(in_w) / in_w_t) * in_w_t + (cur_filter_s - 1))
      out_h_hw = int(ceil(float(out_h) / out_h_t) * out_h_t + (nxt_filter_s - 1))
      out_w_hw = int(ceil(float(out_w) / out_w_t) * out_w_t + (nxt_filter_s - 1))

      task_num1 = int(ceil(float(in_num) / in_num_t) * ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
      task_num2 = int(ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
      if layer_type == "separable_conv":
        local_accum_num = int(in_num_t / SA_SIMD * 1 * 1)
      else:
        local_accum_num = int(in_num_t / SA_SIMD * filter_s * filter_s)
#      local_accum_num = int(in_num_t / SA_SIMD * filter_s * filter_s)
      local_reg_num = int((in_h_t / stride) * (in_w_t / SA_COLS / stride) * (out_num_t / SA_ROWS))
      row_il_factor = int(out_num_t / SA_ROWS)
      col_il_factor = int(in_w_t / SA_COLS / stride)

      if layer_name == "MConv_Stage2_L1_5":
        MConv_Stage2_L1_5_out_num = out_num
        MConv_Stage2_L1_5_out_num_hw = out_num_hw
        MConv_Stage2_L1_5_out_h = out_h
        MConv_Stage2_L1_5_out_w = out_w

      if layer_name == "MConv_Stage2_L2_5":
        Mconv_Stage2_L2_5_out_num = out_num
        Mconv_Stage2_L2_5_out_num_hw = out_num_hw
        Mconv_Stage2_L2_5_out_h = out_h
        Mconv_Stage2_L2_5_out_w = out_w

      # calculate cin_ cout, weight, bias size
      cin_size = in_num * in_h * in_w
      cin_size_hw = in_num_hw * in_h_hw * in_w_hw

      cout_size = out_num * out_h * out_w
      cout_size_hw = out_num_hw * out_h_hw * out_w_hw
      cout_size_hw_concat = out_num * out_h_hw * out_w_hw

      if layer_type == "separable_conv":
        weight_size = in_num * filter_s * filter_s + in_num * out_num * 1 * 1
        weight_size_hw = in_num_hw * filter_s * filter_s + in_num_hw *out_num_hw * 1 * 1
        bias_size = out_num
        bias_size_hw = out_num_hw
      elif layer_type == "convb":
        weight_size = in_num * out_num * filter_s * filter_s
        weight_size_hw = in_num_hw * out_num_hw * filter_s * filter_s
        bias_size = out_num
        bias_size_hw = out_num_hw
      else: # maxpool
        weight_size = 0
        weight_size_hw = 0
        bias_size = 0
        bias_size_hw = 0

      weight_load_size = weight_size_hw
      bias_load_size = bias_size_hw
      weight_load.write(str(weight_offset) + " " + str(weight_load_size) + '\n')
      bias_load.write(str(bias_offset) + " " + str(bias_load_size) + '\n')

      weight_offset = weight_offset + weight_load_size
      bias_offset = bias_offset + bias_load_size

      layer_cin_size[layer_name] = cin_size
      layer_cin_size_hw[layer_name] = cin_size_hw

      layer_cout_size[layer_name] = cout_size
      layer_cout_size_hw[layer_name] = cout_size_hw
      layer_cout_size_hw_concat[layer_name] = cout_size_hw_concat

      layer_weight_size[layer_name] = weight_size
      layer_weight_size_hw[layer_name] = weight_size_hw

      layer_bias_size[layer_name] = bias_size
      layer_bias_size_hw[layer_name] = bias_size_hw

      layer_config = {}
      layer_config['IN_NUM'] = in_num
      layer_config['OUT_NUM'] = out_num
      layer_config['IN_H'] = in_h
      layer_config['IN_W'] = in_w
      layer_config['OUT_H'] = out_h
      layer_config['OUT_W'] = out_w
      layer_config['IN_NUM_HW'] = in_num_hw
      layer_config['OUT_NUM_HW'] = out_num_hw
      layer_config['IN_H_HW'] = in_h_hw
      layer_config['IN_W_HW'] = in_w_hw
      layer_config['OUT_H_HW'] = out_h_hw
      layer_config['OUT_W_HW'] = out_w_hw
      if layer_type == "separable_conv":
        layer_config['FILTER_S1'] = filter_s
        layer_config['FILTER_S2'] = 1
      elif layer_type == "convb":
        layer_config['FILTER_S1'] = 1
        layer_config['FILTER_S2'] = filter_s
      elif layer_type == "max_pool":
        layer_config['FILTER_S1'] = 1
        layer_config['FILTER_S2'] = 1

      layer_config['STRIDE'] = stride
      layer_config['IN_NUM_T'] = in_num_t
      layer_config['OUT_NUM_T'] = out_num_t
      layer_config['IN_H_T'] = in_h_t
      layer_config['IN_W_T'] = in_w_t
      layer_config['NXT_LAYER_BATCH'] = 1
      layer_config['TASK_NUM1'] = task_num1
      layer_config['TASK_NUM2'] = task_num2
      layer_config['LOCAL_ACCUM_NUM'] = local_accum_num
      layer_config['LOCAL_REG_NUM'] = local_reg_num
      layer_config['ROW_IL_FACTOR'] = row_il_factor
      layer_config['COL_IL_FACTOR'] = col_il_factor

      if layer_type == "separable_conv":
        depth_conv_en = 1
        conv_en = 1
      elif layer_type == "convb":
        depth_conv_en = 0
        conv_en = 1
      elif layer_type == "max_pool":
        depth_conv_en = 0
        conv_en = 0

      up_sample_en = 0
#      ld_sel = 0
#      wr_sel = 0
      inter_load_en = 0
      inter_write_en = 0

#      if stage2_layer_cnt == 0:
#        inter_write_en = 1
#        inter_load_en = 0
#      elif stage2_layer_cnt == STAGE2_LAYERS - 1:
#        inter_write_en = 0
#        inter_load_en = 1
#      else:
#        inter_write_en = 1
#        inter_load_en = 1

      if stage2_layer_cnt < STAGE2_LAYERS - 1:
        if stage2_channel_cnt == 1:
          layer_config['NXT_LAYER_BATCH'] = 2
          max_layer_batch = max(max_layer_batch, 2)

      layer_en = depth_conv_en + (conv_en << 1) + (relu_en << 2) + (pool_en << 3) + (up_sample_en << 4) + (bias_en << 5) + (inter_load_en << 6) + (inter_write_en << 7)

      layer_config['LAYER_EN'] = layer_en
      layer_configs[layer_name] = layer_config

      if stage2_layer_cnt == STAGE2_LAYERS - 1 and stage2_channel_cnt == 0 and stage2_iter_cnt == 0:
        macros.write("#define STAGE2L_OUT_NUM " + str(out_num) + '\n')
        macros.write("#define STAGE2L_OUT_NUM_T " + str(out_num_t) + '\n')
        macros.write("#define STAGE2L_OUT_H " + str(out_h) + '\n')
        macros.write("#define STAGE2L_OUT_W " + str(out_w) + '\n')
        macros.write("#define STAGE2L_OUT_NUM_HW " + str(int(out_num_hw)) + '\n')
        macros.write("#define STAGE2L_OUT_H_HW " + str(int(out_h_hw)) + '\n')
        macros.write("#define STAGE2L_OUT_W_HW " + str(int(out_w_hw)) + '\n')
        macros.write("#define STAGE2L_K " + str(int(nxt_filter_s)) + '\n')

      if stage2_layer_cnt == STAGE2_LAYERS - 1 and stage2_channel_cnt == 1 and stage2_iter_cnt == 0:
        macros.write("#define STAGE2R_OUT_NUM " + str(out_num) + '\n')
        macros.write("#define STAGE2R_OUT_NUM_T " + str(out_num_t) + '\n')
        macros.write("#define STAGE2R_OUT_H " + str(out_h) + '\n')
        macros.write("#define STAGE2R_OUT_W " + str(out_w) + '\n')
        macros.write("#define STAGE2R_OUT_NUM_HW " + str(int(out_num_hw)) + '\n')
        macros.write("#define STAGE2R_OUT_H_HW " + str(int(out_h_hw)) + '\n')
        macros.write("#define STAGE2R_OUT_W_HW " + str(int(out_w_hw)) + '\n')
        macros.write("#define STAGE2R_K " + str(int(nxt_filter_s)) + '\n')

      stage2_layer_cnt = stage2_layer_cnt + 1
      if stage2_layer_cnt == STAGE2_LAYERS:
        stage2_layer_cnt = 0
        stage2_channel_cnt = stage2_channel_cnt + 1
        if stage2_channel_cnt == 2:
          stage2_channel_cnt = 0
          stage2_iter_cnt = stage2_iter_cnt + 1
          if stage2_iter_cnt == STAGE2_ITER:
            stage2_iter_cnt = 0
            current_model = "STAGE2"
            break
          else:
            line_id = stage2_line_id - 1

    line_id = line_id + 1

  # calculate the region size
  region0_layers = ["Conv2d_0", "Conv2d_1", "Conv2d_2", "Conv2d_3", "Conv2d_4", "Conv2d_5", "Conv2d_6"]
  region1_layers = ["Conv2d_8", "Conv2d_9", "Conv2d_10"]
  region2_layers = ["MConv_Stage1_L1_1", "MConv_Stage1_L1_2", "MConv_Stage1_L1_3", "MConv_Stage1_L1_4", \
                    "MConv_Stage1_L2_1", "MConv_Stage1_L2_2", "MConv_Stage1_L2_3", "MConv_Stage1_L2_4"]
  region3_layers = ["MConv_Stage2_L1_1", "MConv_Stage2_L1_2", "MConv_Stage2_L1_3", "MConv_Stage2_L1_4", \
                    "MConv_Stage2_L2_1", "MConv_Stage2_L2_2", "MConv_Stage2_L2_3", "MConv_Stage2_L2_4"]
  region4_layers = ["MConv_Stage1_L1_5", "MConv_Stage1_L2_5", "Conv2d_3_pool", "Conv2d_7", "Conv2d_11"]
  region5_layers = ["MConv_Stage1_L1_5", "MConv_Stage1_L2_5"]

  region0_offset = in_out_offset
  region0_size = 0
  region1_size = 0
  region2_size = 0
  region3_size = 0
  region4_size = 0
  region5_size = 0

  for layer_name in region0_layers:
    region0_size += layer_cout_size_hw[layer_name]
  region1_offset = region0_offset + region0_size
  for layer_name in region1_layers:
    region1_size += layer_cout_size_hw[layer_name]
  region2_offset = region1_offset + region1_size
  for layer_name in region2_layers:
    region2_size += layer_cout_size_hw[layer_name]
  region3_offset = region2_offset + region2_size
  for layer_name in region3_layers:
    region3_size += layer_cout_size_hw[layer_name]
  region4_offset = region3_offset + region3_size
  for layer_name in region4_layers:
    region4_size += layer_cout_size_hw[layer_name]
  region5_offset = region4_offset + region4_size
  for layer_name in region5_layers:
    region5_size += layer_cout_size_hw[layer_name]
#  region4_size += layer_cout_size_hw["MConv_Stage1_L1_5"]
#  region4_size += layer_cout_size_hw["MConv_Stage1_L2_5"]
#  region4_size += layer_cout_size_hw["Conv2d_3_pool"]
#  region4_size += layer_cout_size_hw["Conv2d_7"]
#  region4_size += layer_cout_size_hw["Conv2d_11"]

  cin_size = region5_offset + region5_size
  weight_size = weight_offset
  bias_size = bias_offset

  macros.write("#define CIN_SIZE " + str(int(cin_size)) + '\n')
  macros.write("#define WEIGHT_SIZE " + str(int(weight_size)) + '\n')
  macros.write("#define BIAS_SIZE " + str(int(bias_size)) + '\n')

  # Pass3: To generate offsets
  layer_output_size = []
  layer_output_size_hw = []

  # reinitialize all parameters
  line_id = 1
  current_model = "VGG"
  cin_offset = 0
  cout_offset = 0
  weight_offset = 0
  bias_offset = 0

  vgg_layer_cnt = 0
  vgg_layer_cnt = 0
  stage1_layer_cnt = 0
  stage1_iter_cnt = 0
  stage2_layer_cnt = 0
  stage2_iter_cnt = 0
  stage1_channel_cnt = 0
  stage2_channel_cnt = 0

  in_num_hw = 0
  out_num_hw = 0
  in_h_hw = 0
  in_w_hw = 0
  out_h_hw = 0
  out_w_hw = 0
  filter_s = 0

  in_num = network_in_num
  out_num = network_in_num
  in_h = network_in_h
  in_w = network_in_w
  out_h = network_in_h
  out_w = network_in_w

  while line_id < len(lines):
    line = lines[line_id].strip('\n')
    content = line.split(",")
    if current_model == "VGG":
      layer_name = content[0]
      if layer_name == "Conv2d_7" or layer_name == "Conv2d_11" or layer_name == "Conv2d_3_pool":
        nxt_filter_s = max(filter_list[VGG_LAYERS], filter_list[VGG_LAYERS + STAGE1_LAYERS * 2])
      else:
        nxt_filter_s = filter_list[vgg_layer_cnt + 1]

      if layer_name == "Conv2d_7":
#        print(region4_offset, layer_cout_size_hw['MConv_Stage1_L1_5'], layer_cout_size_hw['MConv_Stage1_L2_5'], layer_cout_size_hw['Conv2d_3_pool'])
        cout_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5'] + layer_cout_size_hw['Conv2d_3_pool'] - in_out_offset
#        print(cout_offset)
      if layer_name == "Conv2d_8":
        cin_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5'] + layer_cout_size_hw['Conv2d_3_pool']
        cout_offset = region1_offset - in_out_offset
      if layer_name == "Conv2d_9":
        cin_offset = region1_offset
      if layer_name == "Conv2d_11":
        cout_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5'] + layer_cout_size_hw['Conv2d_3_pool'] + layer_cout_size_hw['Conv2d_7'] - in_out_offset
      if layer_name == "Conv2d_3_pool":
        cin_offset = region0_offset + layer_cout_size_hw['Conv2d_0'] + layer_cout_size_hw['Conv2d_1'] + layer_cout_size_hw['Conv2d_2'] + OUT_NUM_T * layer_configs['Conv2d_3']['OUT_W_HW'] * int(layer_configs['Conv2d_4']['FILTER_S1'] / 2) + OUT_NUM_T * int(layer_configs['Conv2d_4']['FILTER_S1'] / 2)
#        print(cin_offset)
#        print(layer_configs['Conv2d_4']['FILTER_S1'], layer_configs['Conv2d_3']['OUT_W_HW'])
        cout_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5'] - in_out_offset

      # Debug
#      if layer_name == "Conv2d_2":
#        print(cout_offset, layer_configs[layer_name]['OUT_W_HW'], nxt_filter_s)

      shifted_cout_offset = cout_offset + layer_configs[layer_name]['OUT_NUM_T'] * layer_configs[layer_name]['OUT_W_HW'] * int(nxt_filter_s / 2) + layer_configs[layer_name]['OUT_NUM_T'] * int(nxt_filter_s / 2) + in_out_offset
#      if layer_name == "Conv2d_7":
#        print(shifted_cout_offset)

      layer_configs[layer_name]['SHIFTED_COUT_OFFSET'] = shifted_cout_offset
      layer_configs[layer_name]['CIN_OFFSET'] = cin_offset
      layer_configs[layer_name]['WEIGHT_OFFSET'] = weight_offset
      layer_configs[layer_name]['BIAS_OFFSET'] = bias_offset
      layer_configs[layer_name]['COUT_OFFSET'] = cout_offset

#      inst0 = [layer_configs[layer_name]['IN_NUM_HW'], layer_configs[layer_name]['OUT_NUM_HW'], layer_configs[layer_name]['IN_H_HW'], layer_configs[layer_name]['IN_W_HW'], layer_configs[layer_name]['OUT_H_HW'], layer_configs[layer_name]['OUT_W_HW']]
#      inst1 = [layer_configs[layer_name]['IN_NUM'], layer_configs[layer_name]['OUT_NUM'], layer_configs[layer_name]['IN_H'], layer_configs[layer_name]['IN_W'], layer_configs[layer_name]['OUT_H'], layer_configs[layer_name]['OUT_W']]
#      inst2 = [cin_offset, weight_offset, bias_offset, shifted_cout_offset, layer_configs[layer_name]['FILTER_S1'], layer_configs[layer_name]['FILTER_S2'], layer_configs[layer_name]['STRIDE']]
#      inst3 = [layer_configs[layer_name]['LAYER_EN'], layer_configs[layer_name]['IN_NUM_T'], layer_configs[layer_name]['OUT_NUM_T']]

#      if layer_name == "Conv2d_7":
#        print(inst2)

#      insts.writelines(" ".join(str(int(e)) for e in inst0) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst1) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst2) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst3) + "\n")
#      insts.writelines("\n")

      cin_offset += layer_cin_size_hw[layer_name]
      weight_offset += layer_weight_size_hw[layer_name]
      bias_offset += layer_bias_size_hw[layer_name]
      cout_offset += layer_cout_size_hw[layer_name]

      vgg_layer_cnt = vgg_layer_cnt + 1
  	  # store the start line number of stage 1
      if vgg_layer_cnt == VGG_LAYERS:
        current_model = "STAGE1"
        stage1_line_id = line_id + 1
      line_id = line_id + 1
    elif current_model == "STAGE1":
      layer_name = content[0]

      if layer_name == "MConv_Stage1_L1_5" or layer_name == "MConv_Stage1_L2_5":
        nxt_filter_s = max(filter_list[VGG_LAYERS], filter_list[VGG_LAYERS + STAGE1_LAYERS * 2])
      else:
        nxt_filter_s = filter_list[VGG_LAYERS + stage1_layer_cnt + stage1_channel_cnt * STAGE1_LAYERS + 1]

      if layer_name == "MConv_Stage1_L1_1":
        cin_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5']
        cout_offset = region2_offset - in_out_offset
      if layer_name == "MConv_Stage1_L1_2":
        cin_offset = region2_offset
      if layer_name == "MConv_Stage1_L1_5":
        cout_offset = region4_offset - in_out_offset
      if layer_name == "MConv_Stage1_L2_1":
        cin_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5']
        cout_offset = region2_offset + region2_size / 2 - in_out_offset
      if layer_name == "MConv_Stage1_L2_2":
        cin_offset = region2_offset + region2_size / 2
      if layer_name == "MConv_Stage1_L2_5":
        cout_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] - in_out_offset

      shifted_cout_offset = cout_offset + layer_configs[layer_name]['OUT_NUM_T'] * layer_configs[layer_name]['OUT_W_HW'] * int(nxt_filter_s / 2) + layer_configs[layer_name]['OUT_NUM_T'] * int(nxt_filter_s / 2) + in_out_offset

      layer_configs[layer_name]['SHIFTED_COUT_OFFSET'] = shifted_cout_offset
      layer_configs[layer_name]['CIN_OFFSET'] = cin_offset
      layer_configs[layer_name]['WEIGHT_OFFSET'] = weight_offset
      layer_configs[layer_name]['BIAS_OFFSET'] = bias_offset
      layer_configs[layer_name]['COUT_OFFSET'] = cout_offset

#      inst0 = [layer_configs[layer_name]['IN_NUM_HW'], layer_configs[layer_name]['OUT_NUM_HW'], layer_configs[layer_name]['IN_H_HW'], layer_configs[layer_name]['IN_W_HW'], layer_configs[layer_name]['OUT_H_HW'], layer_configs[layer_name]['OUT_W_HW']]
#      inst1 = [layer_configs[layer_name]['IN_NUM'], layer_configs[layer_name]['OUT_NUM'], layer_configs[layer_name]['IN_H'], layer_configs[layer_name]['IN_W'], layer_configs[layer_name]['OUT_H'], layer_configs[layer_name]['OUT_W']]
#      inst2 = [cin_offset, weight_offset, bias_offset, shifted_cout_offset, layer_configs[layer_name]['FILTER_S1'], layer_configs[layer_name]['FILTER_S2'], layer_configs[layer_name]['STRIDE']]
#      inst3 = [layer_configs[layer_name]['LAYER_EN'], layer_configs[layer_name]['IN_NUM_T'], layer_configs[layer_name]['OUT_NUM_T']]
#
#      insts.writelines(" ".join(str(int(e)) for e in inst0) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst1) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst2) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst3) + "\n")
#      insts.writelines("\n")

      cin_offset += layer_cin_size_hw[layer_name]
      weight_offset += layer_weight_size_hw[layer_name]
      bias_offset += layer_bias_size_hw[layer_name]
      cout_offset += layer_cout_size_hw[layer_name]

## Change the execution order of two branches
#      stage1_channel_cnt = stage1_channel_cnt + 1
#      if stage1_channel_cnt == 2:
#        stage1_channel_cnt = 0
#        stage1_layer_cnt = stage1_layer_cnt + 1
#        if stage1_layer_cnt == STAGE1_LAYERS:
#          stage1_layer_cnt = 0
#          stage1_iter_cnt = stage1_iter_cnt + 1
#          if stage1_iter_cnt == STAGE1_ITER:
#            stage1_iter_cnt = 0
#            stage2_line_id = line_id + 1
#            current_model = "STAGE2"

#      line_id = stage1_line_id + stage1_channel_cnt * STAGE1_LAYERS + stage1_layer_cnt + stage1_iter_cnt * STAGE1_LAYERS * 2
#      if current_model == "STAGE2":
#        line_id = stage2_line_id
#      print(stage1_channel_cnt, stage1_layer_cnt, stage1_iter_cnt)

      stage1_layer_cnt = stage1_layer_cnt + 1
      if stage1_layer_cnt == STAGE1_LAYERS:
        stage1_layer_cnt = 0
        stage1_channel_cnt = stage1_channel_cnt + 1
        if stage1_channel_cnt == 2:
          stage1_channel_cnt = 0
          stage1_iter_cnt = stage1_iter_cnt + 1
          if stage1_iter_cnt == STAGE1_ITER:
            stage1_iter_cnt = 0
            current_model = "STAGE2"
            stage2_line_id = line_id + 1
      line_id = line_id + 1
    elif current_model == "STAGE2":
      layer_name = content[0]

      if layer_name == "MConv_Stage2_L1_5" or layer_name == "MConv_Stage2_L2_5":
        nxt_filter_s = max(filter_list[VGG_LAYERS], filter_list[VGG_LAYERS + STAGE1_LAYERS * 2])
      else:
        nxt_filter_s = filter_list[VGG_LAYERS + STAGE1_LAYERS + stage2_layer_cnt + stage2_channel_cnt * STAGE2_LAYERS + 1]

      if stage2_iter_cnt % 2 == 0: # [2,4,6]
        if layer_name == "MConv_Stage2_L1_1":
          cin_offset = region4_offset
          cout_offset = region3_offset - in_out_offset
        if layer_name == "MConv_Stage2_L1_2":
          cin_offset = region3_offset
        if layer_name == "MConv_Stage2_L1_5":
#          cout_offset = region4_offset - in_out_offset
          cout_offset = region5_offset - in_out_offset
          if stage2_iter_cnt == STAGE2_ITER - 1:
            macros.write("#define STAGE2L_OFFSET " + str(int(cout_offset + in_out_offset)) + '\n')
        if layer_name == "MConv_Stage2_L2_1":
          cin_offset = region4_offset
          cout_offset = region3_offset + region3_size / 2 - in_out_offset
        if layer_name == "MConv_Stage2_L2_2":
          cin_offset = region3_offset + region3_size / 2
        if layer_name == "MConv_Stage2_L2_5":
#          cout_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] - in_out_offset
          cout_offset = region5_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] - in_out_offset
          if stage2_iter_cnt == STAGE2_ITER - 1:
            macros.write("#define STAGE2R_OFFSET " + str(int(cout_offset + in_out_offset)) + '\n')

      elif stage2_iter_cnt % 2 == 1: # [3,5]
        if layer_name == "MConv_Stage2_L1_1":
#          cin_offset = region4_offset
          cin_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5']
          cout_offset = region3_offset - in_out_offset
        if layer_name == "MConv_Stage2_L1_2":
          cin_offset = region3_offset
        if layer_name == "MConv_Stage2_L1_5":
          cout_offset = region4_offset - in_out_offset
          if stage2_iter_cnt == STAGE2_ITER - 1:
            macros.write("#define STAGE2L_OFFSET " + str(int(cout_offset + in_out_offset)) + '\n')
        if layer_name == "MConv_Stage2_L2_1":
#          cin_offset = region4_offset
          cin_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] + layer_cout_size_hw['MConv_Stage1_L2_5']
          cout_offset = region3_offset + region3_size / 2 - in_out_offset
        if layer_name == "MConv_Stage2_L2_2":
          cin_offset = region3_offset + region3_size / 2
        if layer_name == "MConv_Stage2_L2_5":
          cout_offset = region4_offset + layer_cout_size_hw['MConv_Stage1_L1_5'] - in_out_offset
          if stage2_iter_cnt == STAGE2_ITER - 1:
            macros.write("#define STAGE2R_OFFSET " + str(int(cout_offset + in_out_offset)) + '\n')

      #print(layer_name)
      shifted_cout_offset = cout_offset + layer_configs[layer_name]['OUT_NUM_T'] * layer_configs[layer_name]['OUT_W_HW'] * int(nxt_filter_s / 2) + layer_configs[layer_name]['OUT_NUM_T'] * int(nxt_filter_s / 2) + in_out_offset

      layer_configs[layer_name + '_' + str(stage2_iter_cnt)] = {}
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)] = {}
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)] = {}
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)] = {}
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)] = {}
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['SHIFTED_COUT_OFFSET'] = shifted_cout_offset
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['CIN_OFFSET'] = cin_offset
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['WEIGHT_OFFSET'] = weight_offset
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['BIAS_OFFSET'] = bias_offset
      layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['COUT_OFFSET'] = cout_offset

#      inst0 = [layer_configs[layer_name]['IN_NUM_HW'], layer_configs[layer_name]['OUT_NUM_HW'], layer_configs[layer_name]['IN_H_HW'], layer_configs[layer_name]['IN_W_HW'], layer_configs[layer_name]['OUT_H_HW'], layer_configs[layer_name]['OUT_W_HW']]
#      inst1 = [layer_configs[layer_name]['IN_NUM'], layer_configs[layer_name]['OUT_NUM'], layer_configs[layer_name]['IN_H'], layer_configs[layer_name]['IN_W'], layer_configs[layer_name]['OUT_H'], layer_configs[layer_name]['OUT_W']]
#      inst2 = [cin_offset, weight_offset, bias_offset, shifted_cout_offset, layer_configs[layer_name]['FILTER_S1'], layer_configs[layer_name]['FILTER_S2'], layer_configs[layer_name]['STRIDE']]
#      inst3 = [layer_configs[layer_name]['LAYER_EN'], layer_configs[layer_name]['IN_NUM_T'], layer_configs[layer_name]['OUT_NUM_T']]

#      insts.writelines(" ".join(str(int(e)) for e in inst0) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst1) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst2) + "\n")
#      insts.writelines(" ".join(str(int(e)) for e in inst3) + "\n")
#      insts.writelines("\n")

      cin_offset += layer_cin_size_hw[layer_name]
      weight_offset += layer_weight_size_hw[layer_name]
      bias_offset += layer_bias_size_hw[layer_name]
      cout_offset += layer_cout_size_hw[layer_name]

      stage2_layer_cnt = stage2_layer_cnt + 1
      if stage2_layer_cnt == STAGE2_LAYERS:
        stage2_layer_cnt = 0
        stage2_channel_cnt = stage2_channel_cnt + 1
        if stage2_channel_cnt == 2:
          stage2_channel_cnt = 0
          stage2_iter_cnt = stage2_iter_cnt + 1
          if stage2_iter_cnt == STAGE2_ITER:
            stage2_iter_cnt = 0
            current_model = "STAGE2"
            break
          else:
            line_id = stage2_line_id - 1
      line_id = line_id + 1
#    line_id = line_id + 1

  macros.write("#define MAX_LAYER_BATCH " + str(int(max_layer_batch)) + '\n')
  # Pass4: To print out insts
  # reinitialize all parameters
  line_id = 1
  current_model = "VGG"

  vgg_layer_cnt = 0
  vgg_layer_cnt = 0
  stage1_layer_cnt = 0
  stage1_iter_cnt = 0
  stage2_layer_cnt = 0
  stage2_iter_cnt = 0
  stage1_channel_cnt = 0
  stage2_channel_cnt = 0

  while line_id < len(lines):
    line = lines[line_id].strip('\n')
    content = line.split(",")
    if current_model == "VGG":
      layer_name = content[0]

      inst0 = [layer_configs[layer_name]['IN_NUM_HW'], layer_configs[layer_name]['OUT_NUM_HW'], layer_configs[layer_name]['IN_H_HW'], layer_configs[layer_name]['IN_W_HW'], layer_configs[layer_name]['OUT_H_HW'], layer_configs[layer_name]['OUT_W_HW']]
      inst1 = [layer_configs[layer_name]['IN_NUM'], layer_configs[layer_name]['OUT_NUM'], layer_configs[layer_name]['IN_H'], layer_configs[layer_name]['IN_W'], layer_configs[layer_name]['OUT_H'], layer_configs[layer_name]['OUT_W']]
      inst2 = [layer_configs[layer_name]['CIN_OFFSET'], layer_configs[layer_name]['WEIGHT_OFFSET'], layer_configs[layer_name]['BIAS_OFFSET'], layer_configs[layer_name]['SHIFTED_COUT_OFFSET'], layer_configs[layer_name]['FILTER_S1'], layer_configs[layer_name]['FILTER_S2'], layer_configs[layer_name]['STRIDE']]
      inst3 = [layer_configs[layer_name]['LAYER_EN'], layer_configs[layer_name]['IN_NUM_T'], layer_configs[layer_name]['OUT_NUM_T'], layer_configs[layer_name]['IN_H_T'], layer_configs[layer_name]['IN_W_T'], layer_configs[layer_name]['NXT_LAYER_BATCH']]
      inst4 = [layer_configs[layer_name]['TASK_NUM1'], layer_configs[layer_name]['TASK_NUM2'], layer_configs[layer_name]['LOCAL_ACCUM_NUM'], layer_configs[layer_name]['LOCAL_REG_NUM'], layer_configs[layer_name]['ROW_IL_FACTOR'], layer_configs[layer_name]['COL_IL_FACTOR']]

      insts.writelines(" ".join(str(int(e)) for e in inst0) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst1) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst2) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst3) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst4) + "\n")
      insts.writelines("\n")

      vgg_layer_cnt = vgg_layer_cnt + 1
  	  # store the start line number of stage 1
      if vgg_layer_cnt == VGG_LAYERS:
        current_model = "STAGE1"
        stage1_line_id = line_id + 1
      line_id = line_id + 1
    elif current_model == "STAGE1":
      layer_name = content[0]

      inst0 = [layer_configs[layer_name]['IN_NUM_HW'], layer_configs[layer_name]['OUT_NUM_HW'], layer_configs[layer_name]['IN_H_HW'], layer_configs[layer_name]['IN_W_HW'], layer_configs[layer_name]['OUT_H_HW'], layer_configs[layer_name]['OUT_W_HW']]
      inst1 = [layer_configs[layer_name]['IN_NUM'], layer_configs[layer_name]['OUT_NUM'], layer_configs[layer_name]['IN_H'], layer_configs[layer_name]['IN_W'], layer_configs[layer_name]['OUT_H'], layer_configs[layer_name]['OUT_W']]
      inst2 = [layer_configs[layer_name]['CIN_OFFSET'], layer_configs[layer_name]['WEIGHT_OFFSET'], layer_configs[layer_name]['BIAS_OFFSET'], layer_configs[layer_name]['SHIFTED_COUT_OFFSET'], layer_configs[layer_name]['FILTER_S1'], layer_configs[layer_name]['FILTER_S2'], layer_configs[layer_name]['STRIDE']]
      inst3 = [layer_configs[layer_name]['LAYER_EN'], layer_configs[layer_name]['IN_NUM_T'], layer_configs[layer_name]['OUT_NUM_T'], layer_configs[layer_name]['IN_H_T'], layer_configs[layer_name]['IN_W_T'], layer_configs[layer_name]['NXT_LAYER_BATCH']]
      inst4 = [layer_configs[layer_name]['TASK_NUM1'], layer_configs[layer_name]['TASK_NUM2'], layer_configs[layer_name]['LOCAL_ACCUM_NUM'], layer_configs[layer_name]['LOCAL_REG_NUM'], layer_configs[layer_name]['ROW_IL_FACTOR'], layer_configs[layer_name]['COL_IL_FACTOR']]

      insts.writelines(" ".join(str(int(e)) for e in inst0) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst1) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst2) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst3) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst4) + "\n")
      insts.writelines("\n")

# Change the execution order of two branches
      stage1_channel_cnt = stage1_channel_cnt + 1
      if stage1_channel_cnt == 2:
        stage1_channel_cnt = 0
        stage1_layer_cnt = stage1_layer_cnt + 1
        if stage1_layer_cnt == STAGE1_LAYERS:
          stage1_layer_cnt = 0
          stage1_iter_cnt = stage1_iter_cnt + 1
          if stage1_iter_cnt == STAGE1_ITER:
            stage1_iter_cnt = 0
            stage2_line_id = line_id + 1
            current_model = "STAGE2"

      line_id = stage1_line_id + stage1_channel_cnt * STAGE1_LAYERS + stage1_layer_cnt + stage1_iter_cnt * STAGE1_LAYERS * 2
      if current_model == "STAGE2":
        line_id = stage2_line_id
#      print(stage1_channel_cnt, stage1_layer_cnt, stage1_iter_cnt)

#      stage1_layer_cnt = stage1_layer_cnt + 1
#      if stage1_layer_cnt == STAGE1_LAYERS:
#        stage1_layer_cnt = 0
#        stage1_channel_cnt = stage1_channel_cnt + 1
#        if stage1_channel_cnt == 2:
#          stage1_channel_cnt = 0
#          stage1_iter_cnt = stage1_iter_cnt + 1
#          if stage1_iter_cnt == STAGE1_ITER:
#            stage1_iter_cnt = 0
#            current_model = "STAGE2"
#            stage2_line_id = line_id + 1
#      line_id = line_id + 1
    elif current_model == "STAGE2":
      layer_name = content[0]

      inst0 = [layer_configs[layer_name]['IN_NUM_HW'], layer_configs[layer_name]['OUT_NUM_HW'], layer_configs[layer_name]['IN_H_HW'], layer_configs[layer_name]['IN_W_HW'], layer_configs[layer_name]['OUT_H_HW'], layer_configs[layer_name]['OUT_W_HW']]
      inst1 = [layer_configs[layer_name]['IN_NUM'], layer_configs[layer_name]['OUT_NUM'], layer_configs[layer_name]['IN_H'], layer_configs[layer_name]['IN_W'], layer_configs[layer_name]['OUT_H'], layer_configs[layer_name]['OUT_W']]
      inst2 = [layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['CIN_OFFSET'], layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['WEIGHT_OFFSET'], layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['BIAS_OFFSET'], layer_configs[layer_name + '_' + str(stage2_iter_cnt)]['SHIFTED_COUT_OFFSET'], layer_configs[layer_name]['FILTER_S1'], layer_configs[layer_name]['FILTER_S2'], layer_configs[layer_name]['STRIDE']]
      inst3 = [layer_configs[layer_name]['LAYER_EN'], layer_configs[layer_name]['IN_NUM_T'], layer_configs[layer_name]['OUT_NUM_T'], layer_configs[layer_name]['IN_H_T'], layer_configs[layer_name]['IN_W_T'], layer_configs[layer_name]['NXT_LAYER_BATCH']]
      inst4 = [layer_configs[layer_name]['TASK_NUM1'], layer_configs[layer_name]['TASK_NUM2'], layer_configs[layer_name]['LOCAL_ACCUM_NUM'], layer_configs[layer_name]['LOCAL_REG_NUM'], layer_configs[layer_name]['ROW_IL_FACTOR'], layer_configs[layer_name]['COL_IL_FACTOR']]

      insts.writelines(" ".join(str(int(e)) for e in inst0) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst1) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst2) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst3) + "\n")
      insts.writelines(" ".join(str(int(e)) for e in inst4) + "\n")
      insts.writelines("\n")

# Change the execution order of two branches
      stage2_channel_cnt = stage2_channel_cnt + 1
      if stage2_channel_cnt == 2:
        stage2_channel_cnt = 0
        stage2_layer_cnt = stage2_layer_cnt + 1
        if stage2_layer_cnt == STAGE2_LAYERS:
          stage2_layer_cnt = 0
          stage2_iter_cnt = stage2_iter_cnt + 1
          if stage2_iter_cnt == STAGE2_ITER:
            stage2_iter_cnt = 0
            current_model = "STAGE2"
            break

      line_id = stage2_line_id + stage2_channel_cnt * STAGE2_LAYERS + stage2_layer_cnt

#      stage2_layer_cnt = stage2_layer_cnt + 1
#      if stage2_layer_cnt == STAGE2_LAYERS:
#        stage2_layer_cnt = 0
#        stage2_channel_cnt = stage2_channel_cnt + 1
#        if stage2_channel_cnt == 2:
#          stage2_channel_cnt = 0
#          stage2_iter_cnt = stage2_iter_cnt + 1
#          if stage2_iter_cnt == STAGE2_ITER:
#            stage2_iter_cnt = 0
#            current_model = "STAGE2"
#            break
#          else:
#            line_id = stage2_line_id - 1
#      line_id = line_id + 1

#    line_id = line_id + 1

  model.close()
  insts.close()

  macros.close()
  weight_load.close()
  bias_load.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Data reorganization.')

  parser.add_argument('-t', '--tile', metavar='TILE', required=True, help='tiling configuration', dest='tile')
  parser.add_argument('-m', '--model', metavar='MODEL', required=True, help='model description', dest='model')
  parser.add_argument('-mc', '--model-config', metavar='MODEL_CONFIG', required=True, help='model topology', dest='model_config')
#  parser.add_argument('--cin', metavar='INPUT_FIGURE', required=True, help='input feature maps', dest='input_figure')
  parser.add_argument('-i', '--input-config', metavar='INPUT_CONFIG', required=True, help='input configuration', dest='input_config')

  args = parser.parse_args()
  run(args.tile, args.model, args.model_config, args.input_config)
