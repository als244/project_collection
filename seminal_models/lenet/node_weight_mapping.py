


## assuming cutting in half feature map
def add_conv_connections(k, i_map, j_map, inp_N, conv_N, map_N, start_map_ind, shared_weight_to_conn):
    center_input = [2 * i_map, 2 * j_map + i_map % 2]
    lower, upper = -conv_N // 2, conv_N // 2
    for inp_row in range(lower, upper + 1):
        for inp_col in range(lower, upper + 1):
            if (center_input[0] + inp_row < 0) or (center_input[0] + inp_row > inp_N - 1) or (center_input[1] + inp_col < 0) or (center_input[1] + inp_col > inp_N - 1):
                inp_connection_ind = 256
            else:
                inp_connection_ind = inp_N * (center_input[0] + inp_row) + center_input[1] + inp_col
            weight_ind = conv_N ** 2 * k + conv_N * inp_row + inp_col
            shared_weight_to_conn[weight_ind] = (inp_connection_ind, start_map_ind + map_N ** 2 * k + map_N * i_map + j_map) 


shared_weight_to_conn = {}

INP_N, CONV_N, MAP_N = 16, 5, 8
START_MAP_IND = 257
for k in range(12):
    for i in range(8):
        for j in range(8):
            add_conv_connections(k, i, j, INP_N, CONV_N, MAP_N, START_MAP_IND, shared_weight_to_conn)

