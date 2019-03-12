from math import ceil
import json
import argparse
from array import array

'''
Functionality: reorganize data
cin[BATCH][H][W][IN_NUM]
bias[OUT_NUM] -> bias[OUT_NUM / OUT_NUM_T][OUT_NUM_T]
weights[K][K][IN_NUM][OUT_NUM] -> weights[OUT_NUM / OUT_NUM_T][OUT_NUM_T][IN_NUM / IN_NUM_T][K][K][IN_NUM_T]
weights[K][K][IN_NUM] -> weights[IN_NUM / IN_NUM_T][K][K][IN_NUM_T]

For weights, the concat layers original sequnece is:
  conv2d_3_pool - conv2d_7 - conv2d_11
we change it to:
  conv2d_7 - conv2d_11 - conv2d_3_pool

The second concat layer original sequence is:
  mconv_stage1_L1_5 - mconv_stage1_L2_5 - conv2d_3_pool - conv2d_7 - conv2d_11
we change it to:
  conv2d_7 - conv2d_11 - conv2d_3_pool - mconv_stage1_L1_5 - mconv_stage1_L2_5
'''

def run(f_tile, f_model, f_model_config, f_input_config, f_weight, f_bias):
  model = open(f_model, "r")
  with open(f_tile, "r") as f:
    tile = json.loads(f.read())
  with open(f_model_config, "r") as f:
    model_config = json.loads(f.read())
  with open(f_input_config, "r") as f:
    input_config = json.loads(f.read())

#  weights = open(f_weight, "r")
#  bias = open(f_bias, "r")

  # load weights, bias
#  with open(f_weight) as f:
#    weights = f.readlines()
#  weights = [x.strip() for x in weights]
#  with open(f_bias) as f:
#    bias = f.readlines()
#  bias = [x.strip() for x in bias]
  with open(f_weight, 'rb') as f:
    weights_arr = array('f')
    weights_arr.frombytes(f.read())
  weights = weights_arr.tolist()

  with open(f_bias, 'rb') as f:
    bias_arr = array('f')
    bias_arr.frombytes(f.read())
  bias = bias_arr.tolist()

  #print(weights[864])

#  weights_reorg = open('./weight_reorg.dat', 'w')
#  bias_reorg = open('./bias_reorg.dat', 'w')
  weights_reorg = []
  bias_reorg = []

  IN_NUM_T = tile['IN_NUM_T']
  OUT_NUM_T = tile['OUT_NUM_T']
  IN_H_T = tile['IN_H_T']
  IN_W_T = tile['IN_W_T']
  OUT_H_T = tile['OUT_H_T']
  OUT_W_T = tile['OUT_W_T']
  K_T = tile['K_T']

  VGG_LAYERS = model_config['VGG_LAYERS']
  STAGE1_LAYERS = model_config['STAGE1_LAYERS']
  STAGE1_ITER = model_config['STAGE1_ITER']
  STAGE2_LAYERS = model_config['STAGE2_LAYERS']
  STAGE2_ITER = model_config['STAGE2_ITER']

  network_in_num = input_config['IN_NUM']
  network_in_h = input_config['IN_H']
  network_in_w = input_config['IN_W']

  weight_offset = 0
  bias_offset = 0
  weight_reorg_offset = 0
  bias_reorg_offset = 0

  in_num = network_in_num
  out_num = network_in_num
  in_h = network_in_h
  in_w = network_in_w
  out_h = network_in_h
  out_w = network_in_w

  lines = []
  for i in model.readlines():
    lines.append(i)

  line_num = len(lines)

  current_model = "VGG" #{VGG, STAGE1, STAGE2}
  stage1_line_id = 0
  stage2_line_id = 0

  vgg_layer_cnt = 0
  stage1_layer_cnt = 0
  stage1_iter_cnt = 0
  stage2_layer_cnt = 0
  stage2_iter_cnt = 0
  stage1_channel_cnt = 0
  stage2_channel_cnt = 0

  filter_list = []
  for line_id in range(1, line_num):
    line = lines[line_id].strip('\n')
    content = line.split(',')
#    if len(content) > 1 and (content[1] == 'convb' or content[1]) == 'separable_conv':
    if len(content) > 1:
      filter_s = int(content[3])
      filter_list.append(filter_s)

  line_id = 1
  while line_id < line_num:
    line = lines[line_id].strip('\n')
    content = line.split(',')
    if current_model == 'VGG':
      pool_en = 0
      if len(content) > 1 and content[4] == '2':
        pool_en = 1
      layer_name = content[0]
      layer_type = content[1]

      in_num = out_num
      in_h = out_h
      in_w = out_w
      out_num = int(content[2])
      filter_s = int(content[3])
      stride = int(content[4])
      in_num_t = int(content[7])
      out_num_t = int(content[8])

      if layer_name == 'Conv2d_3_pool':
        in_num = Conv2d_3_out_num
        in_h = Conv2d_3_out_h
        in_w = Conv2d_3_out_w

      if pool_en == 1:
        out_h = (in_h + 1) / 2
        out_w = (in_w + 1) / 2
      else:
        out_h = in_h
        out_w = in_w

      cur_filter_s = filter_s
      nxt_filter_s = 0
      if layer_name == 'Conv2d_7' or layer_name == 'Conv2d_11' or layer_name == 'Conv2d_3_pool':
        nxt_filter_s = max(filter_list[VGG_LAYERS], filter_list[VGG_LAYERS + STAGE1_LAYERS * 2])
      else:
        nxt_filter_s = filter_list[vgg_layer_cnt + 1]

      in_num_hw = ceil(float(in_num) / in_num_t) * in_num_t
      out_num_hw = ceil(float(out_num) / out_num_t) * out_num_t
      in_h_hw = ceil(float(in_h) / IN_H_T) * IN_H_T + (cur_filter_s - 1)
      in_w_hw = ceil(float(in_w) / IN_W_T) * IN_W_T + (cur_filter_s - 1)
      out_h_hw = ceil(float(out_h) / OUT_H_T) * OUT_H_T + (nxt_filter_s - 1)
      out_w_hw = ceil(float(out_w) / OUT_W_T) * OUT_W_T + (nxt_filter_s - 1)

      if layer_name == 'Conv2d_3':
        Conv2d_3_out_num = out_num
        Conv2d_3_out_num_hw = out_num_hw
        Conv2d_3_out_h = out_h
        Conv2d_3_out_w = out_w

      if layer_name == 'Conv2d_7':
        Conv2d_7_out_num = out_num
        Conv2d_7_out_num_hw = out_num_hw
        Conv2d_7_out_h = out_h
        Conv2d_7_out_w = out_w

      if layer_name == 'Conv2d_11':
        Conv2d_11_out_num = out_num
        Conv2d_11_out_num_hw = out_num_hw
        Covn2d_11_out_h = out_h
        Conv2d_11_out_w = out_w

      if layer_name == 'Conv2d_3_pool':
        Conv2d_3_pool_out_num = out_num
        Conv2d_3_pool_out_num_hw = out_num_hw
        Conv2d_3_pool_out_h = out_h
        Conv2d_3_pool_out_w = out_w

      weights_layer_raw = []
      weights_layer_reorg = []
      bias_layer_raw = []
      bias_layer_reorg = []

      if layer_type == "separable_conv":
        for weight_idx in range(in_num * filter_s * filter_s):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        for i1 in range(int(ceil(float(in_num) / in_num_t))):
          for p in range(filter_s):
            for q in range(filter_s):
              for i2 in range(in_num_t):
                i = i1 * in_num_t + i2
#                raw_idx = i * filter_s * filter_s + p * filter_s + q
                raw_idx = p * filter_s * in_num + q * in_num + i
                if (i < in_num):
                  weights_layer_reorg.append(weights_layer_raw[raw_idx])
                else:
                  weights_layer_reorg.append(0.0)

        for weight_idx in range(int(in_num_hw * filter_s * filter_s)):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += in_num * filter_s * filter_s
        weight_reorg_offset += in_num_hw * filter_s * filter_s

        #print(weight_offset, weight_reorg_offset)

        weights_layer_raw = []
        weights_layer_reorg = []

        for weight_idx in range(out_num * in_num * 1 * 1):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        #print(weights_layer_raw[0])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for i1 in range(int(ceil(float(in_num) / in_num_t))):
            for o2 in range(out_num_t):
              for p in range(1):
                for q in range(1):
                  for i2 in range(in_num_t):
                    o = o1 * out_num_t + o2
                    i = i1 * in_num_t + i2
                    raw_idx = p * 1 * in_num * out_num + q * in_num * out_num + i * out_num + o
                    if (o < out_num) and (i < in_num):
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    else:
                      weights_layer_reorg.append(0.0)

#        for o1 in range(int(ceil(float(out_num) / OUT_NUM_T))):
#          for o2 in range(OUT_NUM_T):
#            for i1 in range(int(ceil(float(in_num) / IN_NUM_T))):
#              for p in range(1):
#                for q in range(1):
#                  for i2 in range(IN_NUM_T):
#                    o = o1 * OUT_NUM_T + o2
#                    i = i1 * IN_NUM_T + i2
#                    raw_idx = p * 1 * in_num * out_num + q * in_num * out_num + i * out_num + o
#                    if (o < out_num) and (i < in_num):
#                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                    else:
#                      weights_layer_reorg.append(0.0)

        for weight_idx in range(int(out_num_hw * in_num_hw * 1 * 1)):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += out_num * in_num * 1 * 1
        weight_reorg_offset += out_num_hw * in_num_hw * 1 * 1

        for bias_idx in range(out_num):
          bias_layer_raw.append(bias[bias_offset + bias_idx])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for o2 in range(out_num_t):
            o = o1 * out_num_t + o2
            raw_idx = o
            if (o < out_num):
              bias_layer_reorg.append(bias_layer_raw[raw_idx])
            else:
              bias_layer_reorg.append(0.0)

        for bias_idx in range(int(out_num_hw)):
          bias_reorg.append(bias_layer_reorg[bias_idx])

        bias_offset += out_num
        bias_reorg_offset += out_num_hw

#        print(layer_name, weight_reorg_offset, bias_reorg_offset)

      elif layer_type == "convb":
        for weight_idx in range(int(out_num * in_num * filter_s * filter_s)):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for i1 in range(int(ceil(float(in_num) / in_num_t))):
            for o2 in range(out_num_t):
              for p in range(filter_s):
                for q in range(filter_s):
                  for i2 in range(in_num_t):
                    o = o1 * out_num_t + o2
                    i = i1 * in_num_t + i2
                    raw_idx = p * filter_s * in_num * out_num + q * in_num * out_num + i * out_num + o
                    if (o < out_num) and (i < in_num):
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    else:
                      weights_layer_reorg.append(0.0)

#        for o1 in range(int(ceil(float(out_num) / OUT_NUM_T))):
#          for o2 in range(OUT_NUM_T):
#            for i1 in range(int(ceil(float(in_num) / IN_NUM_T))):
#              for p in range(filter_s):
#                for q in range(filter_s):
#                  for i2 in range(IN_NUM_T):
#                    o = o1 * OUT_NUM_T + o2
#                    i = i1 * IN_NUM_T + i2
#                    raw_idx = p * filter_s * in_num * out_num + q * in_num * out_num + i * out_num + o
#                    if (o < out_num) and (i < in_num):
#                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                    else:
#                      weights_layer_reorg.append(0.0)

        for weight_idx in range(int(out_num_hw * in_num_hw * filter_s * filter_s)):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += out_num * in_num * filter_s * filter_s
        weight_reorg_offset += out_num_hw * in_num_hw * filter_s * filter_s

        for bias_idx in range(out_num):
          bias_layer_raw.append(bias[bias_offset + bias_idx])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for o2 in range(out_num_t):
            o = o1 * out_num_t + o2
            raw_idx = o
            if (o < out_num):
              bias_layer_reorg.append(bias_layer_raw[raw_idx])
            else:
              bias_layer_reorg.append(0.0)

#        print(out_num, out_num_hw, len(bias_layer_reorg))

        for bias_idx in range(int(out_num_hw)):
          bias_reorg.append(bias_layer_reorg[bias_idx])

        bias_offset += out_num
        bias_reorg_offset += out_num_hw

        print(layer_name, weight_reorg_offset, bias_reorg_offset)

      vgg_layer_cnt = vgg_layer_cnt + 1
      if vgg_layer_cnt == VGG_LAYERS:
        current_model = "STAGE1"
    elif current_model == 'STAGE1':
      pool_en = 0
      if len(content) > 1 and content[4] == '2':
        pool_en = 1
      layer_name = content[0]
      layer_type = content[1]

      in_num = out_num
      in_h = out_h
      in_w = out_w
      out_num = int(content[2])
      filter_s = int(content[3])
      stride = int(content[4])
      in_num_t = int(content[7])
      out_num_t = int(content[8])

      if stage1_layer_cnt == 0:
        in_num = Conv2d_3_pool_out_num + Conv2d_7_out_num + Conv2d_11_out_num
        in_h = Conv2d_3_pool_out_h
        in_w = Conv2d_3_pool_out_w

      if pool_en == 1:
        out_h = (in_h + 1) / 2
        out_w = (in_w + 1) / 2
      else:
        out_h = in_h
        out_w = in_w

      cur_filter_s = filter_s
      nxt_filter_s = 0
      if layer_name == "MConv_Stage1_L1_5" or layer_name == "MConv_Stage1_L2_5":
        nxt_filter_s = filter_list[VGG_LAYERS + STAGE1_LAYERS * 2]
      else:
        nxt_filter_s = filter_list[VGG_LAYERS + stage1_layer_cnt + stage1_channel_cnt * STAGE1_LAYERS + 1]

      in_num_hw = ceil(float(in_num) / in_num_t) * in_num_t
      out_num_hw = ceil(float(out_num) / out_num_t) * out_num_t
      in_h_hw = ceil(float(in_h) / IN_H_T) * IN_H_T + (cur_filter_s - 1)
      in_w_hw = ceil(float(in_w) / IN_W_T) * IN_W_T + (cur_filter_s - 1)
      out_h_hw = ceil(float(out_h) / OUT_H_T) * OUT_H_T + (nxt_filter_s - 1)
      out_w_hw = ceil(float(out_w) / OUT_W_T) * OUT_W_T + (nxt_filter_s - 1)

      if layer_name == 'MConv_Stage1_L1_1' or layer_name == 'MConv_Stage1_L2_1':
        in_num_hw = Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw + Conv2d_11_out_num_hw

      if layer_name == 'MConv_Stage1_L1_5':
        MConv_Stage1_L1_5_out_num = out_num
        MConv_Stage1_L1_5_out_num_hw = out_num_hw
        MConv_Stage1_L1_5_out_h = out_h
        MConv_Stage1_L1_5_out_w = out_w

      if layer_name == "MConv_Stage1_L2_5":
        MConv_Stage1_L2_5_out_num = out_num
        MConv_Stage1_L2_5_out_num_hw = out_num_hw
        MConv_Stage1_L2_5_out_h = out_h
        MConv_Stage1_L2_5_out_w = out_w

      weights_layer_raw = []
      weights_layer_reorg = []
      bias_layer_raw = []
      bias_layer_reorg = []

      if layer_type == "separable_conv":
#        if layer_name == "MConv_Stage1_L1_1" or layer_name == "MConv_Stage1_L2_1":
#          for weight_idx in range(Conv2d_7_out_num * filter_s * filter_s):
#            weights_layer_raw.append(weights[weight_offset + Conv2d_3_pool_out_num * filter_s * filter_s + weight_idx])
#          for weight_idx in range(Conv2d_11_out_num * filter_s * filter_s):
#            weights_layer_raw.append(weights[weight_offset + (Conv2d_3_pool_out_num + Conv2d_7_out_num) * filter_s * filter_s + weight_idx])
#          for weight_idx in range(Conv2d_3_pool_out_num * filter_s * filter_s):
#            weights_layer_raw.append(weights[weight_offset + weight_idx])
#        else:
#          for weight_idx in range(in_num * filter_s * filter_s):
#            weights_layer_raw.append(weights[weight_offset + weight_idx])

        for weight_idx in range(in_num * filter_s * filter_s):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        if layer_name == "MConv_Stage1_L1_1" or layer_name == "MConv_Stage1_L2_1":
          for i1 in range(int(ceil(in_num_hw / in_num_t))):
            for p in range(filter_s):
              for q in range(filter_s):
                for i2 in range(in_num_t):
                  i = i1 * in_num_t + i2
                  if (i < Conv2d_3_pool_out_num):
                    raw_idx = int(p * filter_s * in_num + q * in_num + i)
                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
                  elif (i >= Conv2d_3_pool_out_num_hw) and (i < Conv2d_3_pool_out_num_hw + Conv2d_7_out_num):
                    raw_idx = int(p * filter_s * in_num + q * in_num + (i - Conv2d_3_pool_out_num_hw + Conv2d_3_pool_out_num))
                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
                  elif (i >= Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw) and (i < Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw + Conv2d_11_out_num):
                    raw_idx = int(p * filter_s * in_num + q * in_num + (i - Conv2d_3_pool_out_num_hw - Conv2d_7_out_num_hw + Conv2d_3_pool_out_num + Conv2d_7_out_num))
                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
                  else:
                    weights_layer_reorg.append(0.0)

#          for i1 in range(int(ceil(Conv2d_3_pool_out_num / IN_NUM_T))):
#            for p in range(filter_s):
#              for q in range(filter_s):
#                for i2 in range(IN_NUM_T):
#                  i = i1 * IN_NUM_T + i2
##                  raw_idx = i * filter_s * filter_s + p * filter_s + q
#                  raw_idx = p * filter_s * in_num + q * in_num + i
#                  if (i < Conv2d_3_pool_out_num):
#                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                  else:
#                    weights_layer_reorg.append('0')
#          for i1 in range(int(ceil(Conv2d_7_out_num / IN_NUM_T))):
#            for p in range(filter_s):
#              for q in range(filter_s):
#                for i2 in range(IN_NUM_T):
#                  i = i1 * IN_NUM_T + i2
##                  raw_idx = (i + Conv2d_3_pool_out_num) * filter_s * filter_s + p * filter_s + q
#                  raw_idx = p * filter_s * in_num + q * in_num + (i + Conv2d_3_pool_out_num)
#                  if (i < Conv2d_7_out_num):
#                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                  else:
#                    weights_layer_reorg.append('0')
#          for i1 in range(int(ceil(Conv2d_11_out_num / IN_NUM_T))):
#            for p in range(filter_s):
#              for q in range(filter_s):
#                for i2 in range(IN_NUM_T):
#                  i = i1 * IN_NUM_T + i2
##                  raw_idx = (i + Conv2d_3_pool_out_num + Conv2d_7_out_num) * filter_s * filter_s + p * filter_s + q
#                  raw_idx = p * filter_s * in_num + q * in_num + (i + Conv2d_3_pool_out_num + Conv2d_7_out_num)
#                  if (i < Conv2d_11_out_num):
#                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                  else:
#                    weights_layer_reorg.append('0')
        else:
          for i1 in range(int(ceil(in_num / in_num_t))):
            for p in range(filter_s):
              for q in range(filter_s):
                for i2 in range(in_num_t):
                  i = i1 * in_num_t + i2
#                  raw_idx = i * filter_s * filter_s + p * filter_s + q
                  raw_idx = p * filter_s * in_num + q * in_num + i
                  if (i < in_num):
                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
                  else:
                    weights_layer_reorg.append(0.0)

        for weight_idx in range(int(in_num_hw * filter_s * filter_s)):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += in_num * filter_s * filter_s
        weight_reorg_offset += in_num_hw * filter_s * filter_s

        weights_layer_raw = []
        weights_layer_reorg = []

#        if layer_name == "MConv_Stage1_L1_1" or layer_name == "MConv_Stage1_L2_1":
#          for o in range(out_num):
#            for i in range(Conv2d_3_pool_out_num * 1 * 1, (Conv2d_3_pool_out_num + Conv2d_7_out_num) * 1 * 1):
#              weights_layer_raw.append(weights[weight_offset + i + o * in_num * 1 * 1])
#            for i in range((Conv2d_3_pool_out_num + Conv2d_7_out_num) * 1 * 1, (Conv2d_3_pool_out_num + Conv2d_7_out_num + Conv2d_11_out_num) * 1 * 1):
#              weights_layer_raw.append(weights[weight_offset + i + o * in_num * 1 * 1])
#            for i in range(Conv2d_3_pool_out_num * 1 * 1):
#              weights_layer_raw.append(weights[weight_offset + i + o * in_num * 1 * 1])
#        else:
#          for weight_idx in range(in_num * out_num * 1 * 1):
#            weights_layer_raw.append(weights[weight_offset + weight_idx])
        for weight_idx in range(in_num * out_num * 1 * 1):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        if layer_name == "MConv_Stage1_L1_1" or layer_name == "MConv_Stage1_L2_1":
          for o1 in range(int(ceil(float(out_num_hw) / out_num_t))):
            for i1 in range(int(ceil(in_num_hw / in_num_t))):
              for o2 in range(out_num_t):
                for p in range(1):
                  for q in range(1):
                    for i2 in range(in_num_t):
                      o = o1 * out_num_t + o2
                      i = i1 * in_num_t + i2
                      if (i < Conv2d_3_pool_out_num) and (o < out_num):
                        raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + i * out_num + o)
                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
                      elif (i >= Conv2d_3_pool_out_num_hw) and (i < Conv2d_3_pool_out_num_hw + Conv2d_7_out_num) and (o < out_num):
                        raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - Conv2d_3_pool_out_num_hw + Conv2d_3_pool_out_num) * out_num + o)
                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
                      elif (i >= Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw) and (i < Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw + Conv2d_11_out_num) and (o < out_num):
                        raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - Conv2d_3_pool_out_num_hw - Conv2d_7_out_num_hw + Conv2d_3_pool_out_num + Conv2d_7_out_num) * out_num + o)
                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
                      else:
                        weights_layer_reorg.append(0.0)

#          for o1 in range(int(ceil(float(out_num) / OUT_NUM_T))):
#            for o2 in range(OUT_NUM_T):
#              for i1 in range(int(ceil(float(Conv2d_3_pool_out_num) / IN_NUM_T))):
#                for p in range(1):
#                  for q in range(1):
#                    for i2 in range(IN_NUM_T):
#                      o = o1 * OUT_NUM_T + o2
#                      i = i1 * IN_NUM_T + i2
##                      raw_idx = o * in_num * 1 * 1 + i * 1 * 1 + p * 1 + q
#                      raw_idx = p * 1 * in_num * out_num + q * in_num * out_num + i * out_num + o
#                      if (o < out_num) and (i < Conv2d_3_pool_out_num):
#                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                      else:
#                        weights_layer_reorg.append('0')
#          for o1 in range(int(ceil(float(out_num) / OUT_NUM_T))):
#            for o2 in range(OUT_NUM_T):
#              for i1 in range(int(ceil(float(Conv2d_7_out_num) / IN_NUM_T))):
#                for p in range(1):
#                  for q in range(1):
#                    for i2 in range(IN_NUM_T):
#                      o = o1 * OUT_NUM_T + o2
#                      i = i1 * IN_NUM_T + i2
##                      raw_idx = o * in_num * 1 * 1 + (i + Conv2d_3_pool_out_num) * 1 * 1 + p * 1 + q
#                      raw_idx = p * 1 * in_num * out_num + q * in_num * out_num + (i + Conv2d_3_pool_out_num) * out_num + o
#                      if (o < out_num) and (i < Conv2d_7_out_num):
#                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                      else:
#                        weights_layer_reorg.append('0')
#          for o1 in range(int(ceil(float(out_num) / OUT_NUM_T))):
#            for o2 in range(OUT_NUM_T):
#              for i1 in range(int(ceil(float(Conv2d_11_out_num) / IN_NUM_T))):
#                for p in range(1):
#                  for q in range(1):
#                    for i2 in range(IN_NUM_T):
#                      o = o1 * OUT_NUM_T + o2
#                      i = i1 * IN_NUM_T + i2
##                      raw_idx = o * in_num * 1 * 1 + (i + Conv2d_3_pool_out_num + Conv2d_7_out_num) * 1 * 1 + p * 1 + q
#                      raw_idx = p * 1 * in_num * out_num + q * in_num * out_num + (i + Conv2d_3_pool_out_num + Conv2d_7_out_num) * out_num + o
#                      if (o < out_num) and (i < Conv2d_11_out_num):
#                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
#                      else:
#                        weights_layer_reorg.append('0')

        else:
          for o1 in range(int(ceil(float(out_num) / out_num_t))):
            for i1 in range(int(ceil(float(in_num) / in_num_t))):
              for o2 in range(out_num_t):
                for p in range(1):
                  for q in range(1):
                    for i2 in range(in_num_t):
                      o = o1 * out_num_t + o2
                      i = i1 * in_num_t + i2
#                      raw_idx = o * in_num * 1 * 1 + i * 1 * 1 + p * 1 + q
                      raw_idx = p * 1 * in_num * out_num + q * in_num * out_num + i * out_num + o
                      if (o < out_num) and (i < in_num):
                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
                      else:
                        weights_layer_reorg.append(0.0)

        for weight_idx in range(int(out_num_hw * in_num_hw * 1 * 1)):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += out_num * in_num * 1 * 1
        weight_reorg_offset += out_num_hw * in_num_hw * 1 * 1

        for bias_idx in range(out_num):
          bias_layer_raw.append(bias[bias_offset + bias_idx])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for o2 in range(out_num_t):
            o = o1 * out_num_t + o2
            raw_idx = o
            if (o < out_num):
              bias_layer_reorg.append(bias_layer_raw[raw_idx])
            else:
              bias_layer_reorg.append(0.0)

        for bias_idx in range(int(out_num_hw)):
          bias_reorg.append(bias_layer_reorg[bias_idx])

        bias_offset += out_num
        bias_reorg_offset += out_num_hw

#        print(layer_name, weight_reorg_offset, bias_reorg_offset)

      elif layer_type == "convb":
        for weight_idx in range(out_num * in_num * filter_s * filter_s):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        for o1 in range(int(ceil(out_num / out_num_t))):
          for i1 in range(int(ceil(in_num / in_num_t))):
            for o2 in range(out_num_t):
              for p in range(filter_s):
                for q in range(filter_s):
                  for i2 in range(in_num_t):
                    o = o1 * out_num_t + o2
                    i = i1 * in_num_t + i2
#                    raw_idx = o * in_num * filter_s * filter_s + i * filter_s * filter_s + p * filter_s + q
                    raw_idx = p * filter_s * in_num * out_num + q * in_num * out_num + i * out_num + o
                    if (o < out_num) and (i < in_num):
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    else:
                      weights_layer_reorg.append(0.0)

        for weight_idx in range(out_num_hw * in_num_hw * filter_s * filter_s):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += out_num * in_num * filter_s * filter_s
        weight_reorg_offset += out_num_hw * in_num_hw * filter_s * filter_s

        for bias_idx in range(out_num):
          bias_layer_raw.append(bias[bias_offset + bias_idx])

        for o1 in range(int(ceil(out_num / out_num_t))):
          for o2 in range(out_num_t):
            o = o1 * out_num_t + o2
            raw_idx = o
            if (o < out_num):
              bias_layer_reorg.append(bias_layer_raw[raw_idx])
            else:
              bias_layer_reorg.append(0.0)

        for bias_idx in range(out_num_hw):
          bias_reorg.append(bias_layer_reorg[bias_idx])

        bias_offset += out_num
        bias_reorg_offset += out_num_hw

        print(layer_name, weight_reorg_offset, bias_reorg_offset)

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
    elif current_model == 'STAGE2':
      pool_en = 0
      if len(content) > 1 and content[4] == '2':
        pool_en = 1
      layer_name = content[0]
      layer_type = content[1]

      in_num = out_num
      in_h = out_h
      in_w = out_w
      out_num = int(content[2])
      filter_s = int(content[3])
      stride = int(content[4])
      in_num_t = int(content[7])
      out_num_t = int(content[8])

      if stage2_layer_cnt == 0:
        in_num = MConv_Stage1_L1_5_out_num + MConv_Stage1_L2_5_out_num + Conv2d_3_pool_out_num + Conv2d_7_out_num + Conv2d_11_out_num
        in_h = Conv2d_3_pool_out_h
        in_w = Conv2d_3_pool_out_w

      if pool_en == 1:
        out_h = (in_h + 1) / 2
        out_w = (in_w + 1) / 2
      else:
        out_h = in_h
        out_w = in_w

      cur_filter_s = filter_s
      nxt_filter_s = 0
      if layer_name == "MConv_Stage2_L1_5" or layer_name == "MConv_Stage2_L2_5":
        nxt_filter_s = filter_list[VGG_LAYERS + STAGE1_LAYERS * 2]
      else:
        nxt_filter_s = filter_list[VGG_LAYERS + STAGE1_LAYERS * 2 + stage2_layer_cnt + stage2_channel_cnt * STAGE2_LAYERS + 1]

      in_num_hw = ceil(float(in_num) / in_num_t) * in_num_t
      out_num_hw = ceil(float(out_num) / out_num_t) * out_num_t
      in_h_hw = ceil(float(in_h) / IN_H_T) * IN_H_T + (cur_filter_s - 1)
      in_w_hw = ceil(float(in_w) / IN_W_T) * IN_W_T + (cur_filter_s - 1)
      out_h_hw = ceil(float(out_h) / OUT_H_T) * OUT_H_T + (nxt_filter_s - 1)
      out_w_hw = ceil(float(out_w) / OUT_W_T) * OUT_W_T + (nxt_filter_s - 1)

      if layer_name == "MConv_Stage2_L1_1" or layer_name == "MConv_Stage2_L2_1":
        in_num_hw = MConv_Stage1_L1_5_out_num_hw + MConv_Stage1_L2_5_out_num_hw + Conv2d_3_pool_out_num_hw + Conv2d_7_out_num_hw + Conv2d_11_out_num_hw

      if layer_name == 'MConv_Stage2_L1_5':
        MConv_Stage2_L1_5_out_num = out_num
        MConv_Stage2_L1_5_out_num_hw = out_num_hw
        MConv_Stage2_L1_5_out_h = out_h
        MConv_Stage2_L1_5_out_w = out_w

      if layer_name == "MConv_Stage2_L2_5":
        Mconv_Stage2_L2_5_out_num = out_num
        Mconv_Stage2_L2_5_out_num_hw = out_num_hw
        Mconv_Stage2_L2_5_out_h = out_h
        Mconv_Stage2_L2_5_out_w = out_w

      weights_layer_raw = []
      weights_layer_reorg = []
      bias_layer_raw = []
      bias_layer_reorg = []

      if layer_type == "separable_conv":
        for weight_idx in range(in_num * filter_s * filter_s):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        if layer_name == "MConv_Stage2_L1_1" or layer_name == "MConv_Stage2_L2_1":
          if stage2_iter_cnt % 2 == 0: # [2,4,6]
            for i1 in range(int(ceil(in_num_hw / in_num_t))):
              for p in range(filter_s):
                for q in range(filter_s):
                  for i2 in range(in_num_t):
                    i = i1 * in_num_t + i2
# MConv_Stage1_L1_5
                    i_lower1 = 0
                    i_upper1 = i_lower1 + MConv_Stage1_L1_5_out_num
                    i_offset1 = 0
# MConv_Stage1_L2_5
                    i_lower2 = i_lower1 + MConv_Stage1_L1_5_out_num_hw
                    i_upper2 = i_lower2 + MConv_Stage1_L2_5_out_num
                    i_offset2 = i_offset1 + MConv_Stage1_L1_5_out_num_hw - MConv_Stage1_L1_5_out_num
# Conv2d_3_pool
                    i_lower3 = i_lower2 + MConv_Stage1_L2_5_out_num_hw
                    i_upper3 = i_lower3 + Conv2d_3_pool_out_num
                    i_offset3 = i_offset2 + (MConv_Stage1_L2_5_out_num_hw - MConv_Stage1_L2_5_out_num)
# Conv2d_7
                    i_lower4 = i_lower3 + Conv2d_3_pool_out_num_hw
                    i_upper4 = i_lower4 + Conv2d_7_out_num
                    i_offset4 = i_offset3 + (Conv2d_3_pool_out_num_hw - Conv2d_3_pool_out_num)
# Conv2d_11
                    i_lower5 = i_lower4 + Conv2d_7_out_num_hw
                    i_upper5 = i_lower5 + Conv2d_11_out_num
                    i_offset5 = i_offset4 + (Conv2d_7_out_num_hw - Conv2d_7_out_num)

                    if (i >= i_lower1) and (i < i_upper1):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset1))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower2) and (i < i_upper2):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset2))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower3) and (i < i_upper3):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset3))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower4) and (i < i_upper4):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset4))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower5) and (i < i_upper5):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset5))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    else:
                      weights_layer_reorg.append(0.0)

          elif stage2_iter_cnt % 2 == 1: # [3,5]
            for i1 in range(int(ceil(in_num_hw / in_num_t))):
              for p in range(filter_s):
                for q in range(filter_s):
                  for i2 in range(in_num_t):
                    i = i1 * in_num_t + i2
# Conv2d_3_pool
                    i_lower1 = 0
                    i_upper1 = i_lower1 + Conv2d_3_pool_out_num
                    i_offset1 = 0
# Conv2d_7
                    i_lower2 = i_lower1 + Conv2d_3_pool_out_num_hw
                    i_upper2 = i_lower2 + Conv2d_7_out_num
                    i_offset2 = Conv2d_3_pool_out_num_hw
# Conv2d_11
                    i_lower3 = i_lower2 + Conv2d_7_out_num_hw
                    i_upper3 = i_lower3 + Conv2d_11_out_num
                    i_offset3 = i_offset2 + Conv2d_7_out_num_hw
# MConv_Stage1_L1_5
                    i_lower4 = i_lower3 + Conv2d_11_out_num_hw
                    i_upper4 = i_lower4 + MConv_Stage1_L1_5_out_num
                    i_offset4 = i_offset3 + Conv2d_11_out_num_hw
# MConv_Stage1_L2_5
                    i_lower5 = i_lower4 + MConv_Stage1_L1_5_out_num_hw
                    i_upper5 = i_lower5 + MConv_Stage1_L2_5_out_num
                    i_offset5 = i_offset4 + MConv_Stage1_L1_5_out_num_hw

                    if (i >= i_lower1) and (i < i_upper1):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset1 + MConv_Stage1_L1_5_out_num + MConv_Stage1_L2_5_out_num))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower2) and (i < i_upper2):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset2 + MConv_Stage1_L1_5_out_num + MConv_Stage1_L2_5_out_num + Conv2d_3_pool_out_num))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower3) and (i < i_upper3):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset3 + MConv_Stage1_L1_5_out_num + MConv_Stage1_L2_5_out_num + Conv2d_3_pool_out_num + Conv2d_7_out_num))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower4) and (i < i_upper4):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset4))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    elif (i >= i_lower5) and (i < i_upper5):
                      raw_idx = int(p * filter_s * in_num + q * in_num + (i - i_offset5 + MConv_Stage1_L1_5_out_num))
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    else:
                      weights_layer_reorg.append(0.0)

        else:
          for i1 in range(int(ceil(float(in_num) / in_num_t))):
            for p in range(filter_s):
              for q in range(filter_s):
                for i2 in range(in_num_t):
                  i = i1 * in_num_t + i2
#                  raw_idx = i * filter_s * filter_s + p * filter_s + q
                  raw_idx = p * filter_s * in_num + q * in_num + i
                  if (i < in_num):
                    weights_layer_reorg.append(weights_layer_raw[raw_idx])
                  else:
                    weights_layer_reorg.append(0.0)

        for weight_idx in range(int(in_num_hw * filter_s * filter_s)):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += in_num * filter_s * filter_s
        weight_reorg_offset += in_num_hw * filter_s * filter_s

        weights_layer_raw = []
        weights_layer_reorg = []

        for weight_idx in range(in_num * out_num * 1 * 1):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        if layer_name == "MConv_Stage2_L1_1" or layer_name == "MConv_Stage2_L2_1":
          if stage2_iter_cnt % 2 == 0: # [2,4,6]
            for o1 in range(int(ceil(float(out_num_hw) / out_num_t))):
              for i1 in range(int(ceil(float(in_num_hw) / in_num_t))):
                for o2 in range(out_num_t):
                  for p in range(1):
                    for q in range(1):
                      for i2 in range(in_num_t):
                        o = o1 * out_num_t + o2
                        i = i1 * in_num_t + i2
# MConv_Stage1_L1_5
                        i_lower1 = 0
                        i_upper1 = i_lower1 + MConv_Stage1_L1_5_out_num
                        i_offset1 = 0
# MConv_Stage1_L2_5
                        i_lower2 = i_lower1 + MConv_Stage1_L1_5_out_num_hw
                        i_upper2 = i_lower2 + MConv_Stage1_L2_5_out_num
                        i_offset2 = i_offset1 + MConv_Stage1_L1_5_out_num_hw - MConv_Stage1_L1_5_out_num
# Conv2d_3_pool
                        i_lower3 = i_lower2 + MConv_Stage1_L2_5_out_num_hw
                        i_upper3 = i_lower3 + Conv2d_3_pool_out_num
                        i_offset3 = i_offset2 + (MConv_Stage1_L2_5_out_num_hw - MConv_Stage1_L2_5_out_num)
# Conv2d_7
                        i_lower4 = i_lower3 + Conv2d_3_pool_out_num_hw
                        i_upper4 = i_lower4 + Conv2d_7_out_num
                        i_offset4 = i_offset3 + (Conv2d_3_pool_out_num_hw - Conv2d_3_pool_out_num)
# Conv2d_11
                        i_lower5 = i_lower4 + Conv2d_7_out_num_hw
                        i_upper5 = i_lower5 + Conv2d_11_out_num
                        i_offset5 = i_offset4 + (Conv2d_7_out_num_hw - Conv2d_7_out_num)

                        if (i >= i_lower1) and (i < i_upper1) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset1) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower2) and (i < i_upper2) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset2) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower3) and (i < i_upper3) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset3) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower4) and (i < i_upper4) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset4) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower5) and (i < i_upper5) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset5) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        else:
                          weights_layer_reorg.append(0.0)

          elif stage2_iter_cnt % 2 == 1: # [3,5]
            for o1 in range(int(ceil(float(out_num_hw) / out_num_t))):
              for i1 in range(int(ceil(float(in_num_hw) / in_num_t))):
                for o2 in range(out_num_t):
                  for p in range(1):
                    for q in range(1):
                      for i2 in range(in_num_t):
                        o = o1 * out_num_t + o2
                        i = i1 * in_num_t + i2
# Conv2d_3_pool
                        i_lower1 = 0
                        i_upper1 = i_lower1 + Conv2d_3_pool_out_num
                        i_offset1 = 0
# Conv2d_7
                        i_lower2 = i_lower1 + Conv2d_3_pool_out_num_hw
                        i_upper2 = i_lower2 + Conv2d_7_out_num
                        i_offset2 = Conv2d_3_pool_out_num_hw
# Conv2d_11
                        i_lower3 = i_lower2 + Conv2d_7_out_num_hw
                        i_upper3 = i_lower3 + Conv2d_11_out_num
                        i_offset3 = i_offset2 + Conv2d_7_out_num_hw
# MConv_Stage1_L1_5
                        i_lower4 = i_lower3 + Conv2d_11_out_num_hw
                        i_upper4 = i_lower4 + MConv_Stage1_L1_5_out_num
                        i_offset4 = i_offset3 + Conv2d_11_out_num_hw
# MConv_Stage1_L2_5
                        i_lower5 = i_lower4 + MConv_Stage1_L1_5_out_num_hw
                        i_upper5 = i_lower5 + MConv_Stage1_L2_5_out_num
                        i_offset5 = i_offset4 + MConv_Stage1_L1_5_out_num_hw

                        if (i >= i_lower1) and (i < i_upper1) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset1 + MConv_Stage1_L1_5_out_num + MConv_Stage1_L2_5_out_num) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower2) and (i < i_upper2) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset2 + MConv_Stage1_L1_5_out_num + MConv_Stage1_L2_5_out_num + Conv2d_3_pool_out_num) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower3) and (i < i_upper3) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset3 + MConv_Stage1_L1_5_out_num + MConv_Stage1_L2_5_out_num + Conv2d_3_pool_out_num + Conv2d_7_out_num) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower4) and (i < i_upper4) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset4) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        elif (i >= i_lower5) and (i < i_upper5) and (o < out_num):
                          raw_idx = int(p * 1 * in_num * out_num + q * in_num * out_num + (i - i_offset5 + MConv_Stage1_L1_5_out_num) * out_num + o)
                          weights_layer_reorg.append(weights_layer_raw[raw_idx])
                        else:
                          weights_layer_reorg.append(0.0)

        else:
          for o1 in range(int(ceil(float(out_num) / out_num_t))):
            for i1 in range(int(ceil(float(in_num) / in_num_t))):
              for o2 in range(out_num_t):
                for p in range(1):
                  for q in range(1):
                    for i2 in range(in_num_t):
                      o = o1 * out_num_t + o2
                      i = i1 * in_num_t + i2
#                      raw_idx = o * in_num * 1 * 1 + i * 1 * 1 + p * 1 + q
                      raw_idx = p * 1 * in_num * out_num + q * in_num * out_num + i * out_num + o
                      if (o < out_num) and (i < in_num):
                        weights_layer_reorg.append(weights_layer_raw[raw_idx])
                      else:
                        weights_layer_reorg.append(0.0)

        for weight_idx in range(int(out_num_hw * in_num_hw * 1 * 1)):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += out_num * in_num * 1 * 1
        weight_reorg_offset += out_num_hw * in_num_hw * 1 * 1

        for bias_idx in range(out_num):
          bias_layer_raw.append(bias[bias_offset + bias_idx])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for o2 in range(out_num_t):
            o = o1 * out_num_t + o2
            raw_idx = o
            if (o < out_num):
              bias_layer_reorg.append(bias_layer_raw[raw_idx])
            else:
              bias_layer_reorg.append(0.0)

        for bias_idx in range(int(out_num_hw)):
          bias_reorg.append(bias_layer_reorg[bias_idx])

        bias_offset += out_num
        bias_reorg_offset += out_num_hw

        print(layer_name, weight_reorg_offset, bias_reorg_offset)

      elif layer_type == "convb":
        for weight_idx in range(out_num * in_num * filter_s * filter_s):
          weights_layer_raw.append(weights[weight_offset + weight_idx])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for i1 in range(int(ceil(float(in_num) / in_num_t))):
            for o2 in range(out_num_t):
              for p in range(filter_s):
                for q in range(filter_s):
                  for i2 in range(in_num_t):
                    o = o1 * out_num_t + o2
                    i = i1 * in_num_t + i2
#                    raw_idx = o * in_num * filter_s * filter_s + i * filter_s * filter_s + p * filter_s + q
                    raw_idx = p * filter_s * in_num * out_num + q * in_num * out_num + i * out_num + o
                    if (o < out_num) and (i < in_num):
                      weights_layer_reorg.append(weights_layer_raw[raw_idx])
                    else:
                      weights_layer_reorg.append(0.0)

        for weight_idx in range(out_num_hw * in_num_hw * filter_s * filter_s):
          weights_reorg.append(weights_layer_reorg[weight_idx])

        weight_offset += out_num * in_num * filter_s * filter_s
        weight_reorg_offset += out_num_hw * in_num_hw * filter_s * filter_s

        for bias_idx in range(out_num):
          bias_layer_raw.append(bias[bias_offset + bias_idx])

        for o1 in range(int(ceil(float(out_num) / out_num_t))):
          for o2 in range(out_num_t):
            o = o1 * out_num_t + o2
            raw_idx = o
            if (o < out_num):
              bias_layer_reorg.append(bias_layer_raw[raw_idx])
            else:
              bias_layer_reorg.append(0.0)

        for bias_idx in range(out_num_hw):
          bias_reorg.append(bias_layer_reorg[bias_idx])

        bias_offset += out_num
        bias_reorg_offset += out_num_hw

        #print(layer_name, weight_reorg_offset, bias_reorg_offset)

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

  # dump out the reorganized data
#  with open('weight_reorg.dat', 'w') as f:
#    for w in weights_reorg:
#      f.write(w + '\n')
#
#  with open('bias_reorg.dat', 'w') as f:
#    for b in bias_reorg:
#      f.write(b + '\n')
  weights_arr = array('f', weights_reorg)
  with open('weight_reorg.bin', 'wb') as f:
    weights_arr.tofile(f)

  bias_arr = array('f', bias_reorg)
  with open('bias_reorg.bin', 'wb') as f:
    bias_arr.tofile(f)

  model.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Data reorganization.')

  parser.add_argument('-t', '--tile', metavar='TILE', required=True, help='tiling configuration', dest='tile')
  parser.add_argument('-m', '--model', metavar='MODEL', required=True, help='model description', dest='model')
  parser.add_argument('-mc', '--model-config', metavar='MODEL_CONFIG', required=True, help='model topology', dest='model_config')
#  parser.add_argument('--cin', metavar='INPUT_FIGURE', required=True, help='input feature maps', dest='input_figure')
  parser.add_argument('-i', '--input-config', metavar='INPUT_CONFIG', required=True, help='input configuration', dest='input_config')
  parser.add_argument('-w', '--weight', metavar='WEIGHT', required=True, help='weights data', dest='weight')
  parser.add_argument('-b', '--bias', metavar='BIAS', required=True, help='bias data', dest='bias')

  args = parser.parse_args()
  run(args.tile, args.model, args.model_config, args.input_config, args.weight, args.bias)
