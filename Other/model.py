
class WaveletCNN(nn.Module):
    def __init__(self, in_channels, class_nums, levels=2, base_channels=16, dropout_rate=0.3):
   
        super(WaveletCNN, self).__init__()
        self.levels = levels
        
        # Wavelet transform layers
        self.dwt = nn.ModuleList([DWTForward(J=1, mode='zero', wave='db1') for _ in range(levels)])
        
        # Projection and convolution blocks
        self.projections = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # Keep track of output channels for each level
        self.level_channels = []
        
        for i in range(levels):
            # Calculate previous channels for feature concatenation
            prev_channels_sum = sum(self.level_channels)
            
            # Input channels include the low-pass and high-pass wavelet coefficients plus previous features
            input_channels = in_channels + 3 * in_channels + prev_channels_sum
            
            # Exponential scaling: base_channels * 2^i
            output_channels = base_channels * (2 ** i)
            self.level_channels.append(output_channels)

            self.projections.append(nn.Conv2d(input_channels, output_channels, kernel_size=1))
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            ))

        # Segmentation Head
        total_channels = sum(self.level_channels)
        self.seg_head = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(total_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(total_channels // 2, total_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Dynamic upsampling layers
        self.upsample_layers = nn.ModuleList()
        current_channels = total_channels // 2
        
        # Create upsampling layers based on levels
        for i in range(levels):
            out_channels = current_channels // 2 if i < levels - 1 else class_nums
            self.upsample_layers.append(
                nn.ConvTranspose2d(current_channels, out_channels, kernel_size=2, stride=2)
            )
            current_channels = out_channels

    def forward(self, x):
        features = []
        low_pass = x
        
        # Wavelet processing
        for i in range(self.levels):
            low_pass, high_pass = self.dwt[i](low_pass)
            batch_size, channels, height, width = low_pass.shape
            _, high_pass_channels, subbands, _, _ = high_pass[0].shape
            high_pass = high_pass[0].view(batch_size, high_pass_channels * subbands, height, width)

            resized_features = [
                F.interpolate(feat, size=(height, width), mode='bilinear', align_corners=False)
                for feat in features
            ]

            concatenated = torch.cat([low_pass, high_pass] + resized_features, dim=1)
            projected = self.projections[i](concatenated)
            conv_out = self.conv_blocks[i](projected)
            features.append(conv_out)

        # Process features for segmentation
        smallest_height, smallest_width = features[-1].shape[2:]
        resized_features = [
            F.interpolate(feat, size=(smallest_height, smallest_width), mode='bilinear', align_corners=False)
            for feat in features
        ]
        
        out = torch.cat(resized_features, dim=1)
        out = self.seg_head(out)

        # Dynamic upsampling based on number of levels
        for upsample_layer in self.upsample_layers:
            out = upsample_layer(out)

        return out



# Initialize the model
net = WaveletCNN(in_channels=num_components , class_nums=NUM_CLASS)

# Input tensor
input_tensor = torch.randn(1, num_components,PATCH_SIZE, PATCH_SIZE)
print(f"Input tensor shape: {input_tensor.shape}")

# Model summary
summary(
    net,
    input_size=(1, num_components, PATCH_SIZE, PATCH_SIZE),
    col_names=['num_params', 'kernel_size', 'mult_adds', 'input_size', 'output_size'],
    col_width=18,
    row_settings=['var_names'],
    depth=4,
)






# import copy

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F


# NCHW_FORMAT = 'NCHW'
# NHWC_FORMAT = 'NHWC'
# to_NHWC_axis = [0, 2, 3, 1] # NCHW -> NHWC
# to_NCHW_axis = [0, 3, 1, 2] # NHWC -> NCHW

# # DEFAULT_DATA_FORMAT = NCHW_FORMAT
# DEFAULT_DATA_FORMAT = NHWC_FORMAT # should now work faster

# PAD_MODE = 'constant'

# # 2d transform, so power 2
# COEFFS_SCALES_2D_v1 = np.array([
#     1 / np.sqrt(2),
#     np.sqrt(2),
#     np.sqrt(2),
#     np.sqrt(2)
# ], dtype=np.float32) ** 2

# # The same scales allows to get coeffs ranges that are consistent
# COEFFS_SCALES_2D_v2 = np.array([
#     1 / np.sqrt(2) ** 2,
#     1 / np.sqrt(2) ** 2,
#     1 / np.sqrt(2) ** 2,
#     1 / np.sqrt(2) ** 2
# ], dtype=np.float32)

# # 2d transform, so use double power only for LL coeffs
# COEFFS_SCALES_2D_v3 = np.array([
#     1 / np.sqrt(2) ** 2,
#     1 / np.sqrt(2),
#     1 / np.sqrt(2),
#     1 / np.sqrt(2)
# ], dtype=np.float32)

# COEFFS_SCALES_2D_v4 = np.array([
#     1 / np.sqrt(2),
#     1,
#     1,
#     1
# ], dtype=np.float32)

# COEFFS_SCALES_2D_v5 = np.array([
#     1 / np.sqrt(2),
#     1,
#     1,
#     np.sqrt(2)
# ], dtype=np.float32)

# # LL taken from v3, H coeffs from v5
# COEFFS_SCALES_2D_v6 = np.array([
#     1 / np.sqrt(2) ** 2,
#     1,
#     1,
#     np.sqrt(2)
# ], dtype=np.float32)

# COEFFS_SCALES_2D_DICT = {
#     1: COEFFS_SCALES_2D_v1,
#     2: COEFFS_SCALES_2D_v2,
#     3: COEFFS_SCALES_2D_v3,
#     4: COEFFS_SCALES_2D_v4,
#     5: COEFFS_SCALES_2D_v5,
#     6: COEFFS_SCALES_2D_v6
# }

# # 6 is the best for preserving source data range for LL and keeping similar ranges for all H details
# # Found with tests.py with and without normalization fro input
# COEFFS_SCALES_V = 6
# COEFFS_SCALES_2D = torch.from_numpy(COEFFS_SCALES_2D_DICT[COEFFS_SCALES_V])

# DEFAULT_SCALE_1D_COEFFS = True
# DEFAULT_SCALE_2d_COEFFS = True


# # ----- Utils -----

# def scale_into_range(x, target_range):
#     src_range = (x.min(), x.max())
#     src_range_size = src_range[1] - src_range[0]
#     target_range_size = target_range[1] - target_range[0]
#     x = (x - src_range[0]) * target_range_size / src_range_size + target_range[0]
#     return x


# def eval_stats_dict(x, name):
#     if isinstance(x, torch.Tensor):
#         x_np = x.detach().cpu().numpy()
#     else:
#         x_np = x
#     round_digits = 3
#     return {
#         f'{name}_min': round(x_np.min(), round_digits),
#         f'{name}_max': round(x_np.max(), round_digits),
#         f'{name}_mean': round(x_np.mean(), round_digits),
#         f'{name}_abs_mean': round(np.abs(x_np).mean(), round_digits),
#         f'{name}_unit_energy': round(np.sqrt((x_np ** 2).sum()).mean(), round_digits)
#     }


# def eval_stats(x):
#     if isinstance(x, torch.Tensor):
#         x_np = x.detach().cpu().numpy()
#     else:
#         x_np = x
#     round_digits = 3
#     x_min = round(x_np.min(), round_digits)
#     x_max = round(x_np.max(), round_digits)
#     x_mean = round(x_np.mean(), round_digits)
#     x_abs_mean = round(np.abs(x_np).mean(), round_digits)
#     q_delta = 10
#     x_q1 = round(np.percentile(x_np, q=q_delta), round_digits)
#     x_q2 = round(np.percentile(x_np, q=100 - q_delta), round_digits)
#     return x_min, x_max, x_mean, x_abs_mean, x_q1, x_q2

# def test_lifting_scheme(image, kernel, forward_2d_op, backward_2d_op, scale_1d_coefs=True, scale_2d_coefs=True,
#                         data_format=DEFAULT_DATA_FORMAT, print_logs=True):
#     if data_format == NCHW_FORMAT:
#         image = np.transpose(image, (2, 0, 1))

#     input_image = image[None, ...].astype(np.float32)
#     input_image = (input_image / 127.5) - 1.0
#     if print_logs:
#         print(f'Input image min: {input_image.min()}, max: {input_image.max()}')

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     input_image = torch.from_numpy(input_image).to(device)
#     coeffs_scales_2d = COEFFS_SCALES_2D

#     anz_image = forward_2d_op(input_image, kernel,
#                               scale_1d_coeffs=scale_1d_coefs,
#                               scale_2d_coeffs=scale_2d_coefs,
#                               coeffs_scales_2d=coeffs_scales_2d,
#                               data_format=data_format)
#     if print_logs:
#         print(f'Input image shape: {input_image.shape}, anz image shape: {anz_image.shape}')
#     # Apply deepcopy as in-placed ops are used and the same tensor is used later
#     restored_image = backward_2d_op(copy.deepcopy(anz_image), kernel,
#                                     scale_1d_coeffs=scale_1d_coefs,
#                                     scale_2d_coeffs=scale_2d_coefs,
#                                     coeffs_scales_2d=coeffs_scales_2d,
#                                     data_format=data_format)
#     diffs = (input_image - restored_image).detach().cpu().numpy()
#     error = (diffs.flatten() ** 2).mean()

#     anz_image_coeffs = extract_coeffs_from_channels(anz_image, data_format=data_format)
#     scaled_anz_image_coeffs = []
#     scales = eval_stats_dict(input_image, 'src')  # add to this dict coeffs stats later
#     coeffs_names = ['x_LL', 'x_LH', 'X_HL', 'X_HH']
#     for idx, c in enumerate(anz_image_coeffs):
#         if scale_2d_coefs:
#             scaled_c = c
#         else:
#             scaled_c = coeffs_scales_2d[idx] * c
#         name = coeffs_names[idx]
#         vis_c = scale_into_range(c, (0, 1))
#         if print_logs:
#             print(f'{name}: src min = {c.min():.3f}, max = {c.max():.3f}, scaled min = {scaled_c.min():.3f}, scaled max = {scaled_c.max():.3f}')
#         scaled_anz_image_coeffs.append(vis_c)
#         coeffs_scales = eval_stats_dict(c, name)
#         scales = {**scales, **coeffs_scales}
#     vis_anz_image = merge_coeffs_into_spatial(scaled_anz_image_coeffs, data_format=data_format)
#     vis_anz_image = vis_anz_image[0].detach().cpu().numpy()
#     if data_format == NCHW_FORMAT:
#         vis_anz_image = np.transpose(vis_anz_image, (1, 2, 0))
#     vis_anz_image = (255 * vis_anz_image).astype(np.uint8)
#     if print_logs:
#         print(f'Analysis/synthesis error: {error}')
#     return vis_anz_image, error, scales

# def test_lifting_scales(image, name, kernel, forward_2d_op, normalize_input=True, data_format=DEFAULT_DATA_FORMAT,
#                         plot_data=True, plot_hist=True):
#     import matplotlib.pyplot as plt

#     if data_format == NCHW_FORMAT:
#         image = np.transpose(image, (2, 0, 1))

#     input_image = image[None, ...].astype(np.float32)
#     if normalize_input:
#         input_image = (input_image / 127.5) - 1.0

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     input_image = torch.from_numpy(input_image).to(device)

#     stats = {}
#     for scales_v in sorted(list(COEFFS_SCALES_2D_DICT.keys())):
#         coeffs_scales_2d = torch.from_numpy(COEFFS_SCALES_2D_DICT[scales_v])
#         anz_image = forward_2d_op(input_image, kernel,
#                                   scale_1d_coeffs=True,
#                                   scale_2d_coeffs=True,
#                                   coeffs_scales_2d=coeffs_scales_2d,
#                                   data_format=data_format)
#         anz_image_coeffs = extract_coeffs_from_channels(anz_image, data_format=data_format)
#         coeffs_names = ['x_LL', 'x_LH', 'X_HL', 'X_HH']
#         if plot_data:
#             fig, ax = plt.subplots(nrows=5, ncols=1)
#             fig.suptitle(f'Scales_2d v={scales_v}, {name}')
#         data = input_image.detach().cpu().numpy().flatten()
#         label = 'Src image'
#         if plot_data:
#             if plot_hist:
#                 q_delta = 15
#                 range_min = np.percentile(data, q=q_delta)
#                 range_max = np.percentile(data, q=100 - q_delta)
#                 ax[0].hist(data, bins=100, range=(range_min, range_max), density=True, label=label)
#             else:
#                 ax[0].plot(data, label=label)
#             ax[0].legend()
#         scale_stats = []
#         scale_stats.append(eval_stats(data))
#         for idx, c in enumerate(anz_image_coeffs, 1):
#             data = c.detach().cpu().numpy().flatten()
#             scale_stats.append(eval_stats(data))
#             label = coeffs_names[idx - 1]
#             if plot_data:
#                 if plot_hist:
#                     ax[idx].hist(data, bins=100, density=True, label=label)
#                 else:
#                     ax[idx].plot(data, label=label)
#                 ax[idx].legend()
#         df_columns = ['min', 'max', 'mean', 'abs_mean', 'q1', 'q2']
#         stats[scales_v] = pd.DataFrame(scale_stats, columns=df_columns, index=['src'] + coeffs_names)

#     print('Stats:')
#     for k in sorted(list(stats.keys())):
#         df = stats[k]
#         print(f'v={k}:\n{df}')
#     if plot_data:
#         plt.show()


# # ----- Merging/splitting coeffs -----

# def prepare_coeffs_for_1d_op(x, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     assert not (across_cols and across_rows) and (across_cols or across_rows)
#     # o - odd, e - even
#     if data_format == NCHW_FORMAT:
#         if across_cols:
#             # Inputs have shape NCHW and operation is applied across W axis (cols)
#             x_ev_0 = x[:, :, :, 0::2]
#             x_od_0 = x[:, :, :, 1::2]
#         else: # across_rows:
#             # Inputs have shape NCHW and operation is applied across H axis (rows)
#             x_ev_0 = x[:, :, 0::2, :]
#             x_od_0 = x[:, :, 1::2, :]
#     else:  # data_format == NHWC_FORMAT:
#         if across_cols:
#             # Inputs have shape NHWC and operation is applied across W axis (cols)
#             x_ev_0 = x[:, :, 0::2, :]
#             x_od_0 = x[:, :, 1::2, :]
#         else: # across_rows:
#             # Inputs have shape NHWC and operation is applied across H axis (rows)
#             x_ev_0 = x[:, 0::2, :, :]
#             x_od_0 = x[:, 1::2, :, :]
#     return (x_ev_0, x_od_0)


# def prepare_coeffs_for_inv_1d_op(x_coefs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     assert not (across_cols and across_rows) and (across_cols or across_rows)
#     # x_coefs: s, d
#     if data_format == NCHW_FORMAT:
#         _, C, H, W = x_coefs.shape
#         if across_cols:
#             # Inputs have shape NCHW and operation is applied across W axis (cols)
#             s, d = x_coefs[:, :, :, : W // 2], x_coefs[:, :, :, W // 2:]
#         else:  # across_rows:
#             # Inputs have shape NCHW and operation is applied across H axis (rows)
#             s, d = x_coefs[:, :, : H // 2, :], x_coefs[:, :, H // 2:, :]
#     else:  # if data_format == NHWC_FORMAT:
#         _, H, W, C = x_coefs.shape
#         if across_cols:
#             # Inputs have shape NHWC and operation is applied across W axis (cols)
#             s, d = x_coefs[:, :, : W // 2, :], x_coefs[:, :, W // 2:, :]
#         else:  # across_rows:
#             # Inputs have shape NHWC and operation is applied across H axis (rows)
#             s, d = x_coefs[:, : H // 2, :, :], x_coefs[:, H // 2:, :, :]
#     return (s, d)


# def join_coeffs_after_1d_op(coeffs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     assert not (across_cols and across_rows) and (across_cols or across_rows)
#     x_s, x_d = coeffs
#     if across_cols:
#         if data_format == NCHW_FORMAT:
#             # Shapes of x_s and x_d here: [N, C, H, W // 2]
#             x = torch.cat([x_s, x_d], dim=3)
#         else:  # data_format == NHWC_FORMAT:
#             # Shapes of x_s and x_d here: [N, H, W // 2, C]
#             x = torch.cat([x_s, x_d], dim=2)
#     else: # across_rows:
#         if data_format == NCHW_FORMAT:
#             # Shapes of x_s and x_d here: [N, C, H // 2, W]
#             x = torch.cat([x_s, x_d], dim=2)
#         else:  # data_format == NHWC_FORMAT:
#             # Shapes of x_s and x_d here: [N, H // 2, W, C]
#             x = torch.cat([x_s, x_d], dim=1)
#     return x


# def join_coeffs_after_inv_1d_op(coeffs, src_shape, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     assert not (across_cols and across_rows) and (across_cols or across_rows)
#     x_ev_0, x_od_0 = coeffs
#     # Thanks to https://stackoverflow.com/questions/44952886/tensorflow-merge-two-2-d-tensors-according-to-even-and-odd-indices,
#     # answer by P-Gn
#     if data_format == NCHW_FORMAT:
#         _, C, H, W = src_shape
#         if across_cols:
#             # x_od_0 shape: [N, C, H, W // 2], x_ev_0 shape: [N, C, H, W // 2] -> [N, C, H, W // 2, 2] -> [N, C, H, W]
#             x = torch.stack([x_ev_0, x_od_0], dim=4)
#             x = torch.reshape(x, [-1, C, H, 1 * W])
#         else: # across_rows:
#             # x_od_0 shape: [N, C, H // 2, W], x_ev_0 shape: [N, C, H // 2, W] -> [N, C, H // 2, 2, W] -> [N, C, H, W]
#             x = torch.stack([x_ev_0, x_od_0], dim=3)
#             x = torch.reshape(x, [-1, C, 1 * H, W])
#     else:  # data_format == NHWC_FORMAT:
#         # Axis is the next after spatial dim
#         _, H, W, C = src_shape
#         if across_cols:
#             # x_od_0 shape: [N, H, W // 2, C], x_ev_0 shape: [N, H, W // 2, C] -> [N, H, W // 2, 2, C] -> [N, H, W, C]
#             x = torch.stack([x_ev_0, x_od_0], dim=3)
#             x = torch.reshape(x, [-1, H, 1 * W, C])
#         else: # across_rows:
#             # x_od_0 shape: [N, H // 2, W, C], x_ev_0 shape: [N, H // 2, W, C] -> [N, H // 2, 2, W, C] -> [N, H, W, C]
#             x = torch.stack([x_ev_0, x_od_0], dim=2)
#             x = torch.reshape(x, [-1, 1 * H, W, C])
#     return x


# def merge_coeffs_into_channels(x_coeffs, data_format=DEFAULT_DATA_FORMAT):
#     x_LL, x_LH, x_HL, x_HH = x_coeffs
#     if data_format == NCHW_FORMAT:
#         concat_axis = 1
#     else: # if data_format == NHWC_FORMAT:
#         concat_axis = 3
#     return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=concat_axis)


# def extract_coeffs_from_channels(x, data_format=DEFAULT_DATA_FORMAT):
#     if data_format == NCHW_FORMAT:
#         n = x.shape[1] // 4
#         x_LL = x[:, (0 * n) : (1 * n), :, :]
#         x_LH = x[:, (1 * n) : (2 * n), :, :]
#         x_HL = x[:, (2 * n) : (3 * n), :, :]
#         x_HH = x[:, (3 * n) : (4 * n), :, :]
#     else: # if data_format == NHWC_FORMAT:
#         n = x.shape[3] // 4
#         x_LL = x[:, :, :, (0 * n) : (1 * n)]
#         x_LH = x[:, : ,:, (1 * n) : (2 * n)]
#         x_HL = x[:, :, :, (2 * n) : (3 * n)]
#         x_HH = x[:, :, :, (3 * n) : (4 * n)]
#     return x_LL, x_LH, x_HL, x_HH


# def merge_coeffs_into_spatial(x_coeffs, data_format=DEFAULT_DATA_FORMAT):
#     x_LL, x_LH, x_HL, x_HH = x_coeffs
#     if data_format == NCHW_FORMAT:
#         h_axis, v_axis = 2, 3
#     else:  # if data_format == NHWC_FORMAT:
#         h_axis, v_axis = 1, 2
#     x = torch.cat([
#         torch.cat([x_LL, x_LH], dim=h_axis),
#         torch.cat([x_HL, x_HH], dim=h_axis)
#     ], dim=v_axis)
#     return x


# def extract_coeffs_from_spatial(x, data_format=DEFAULT_DATA_FORMAT):
#     if data_format == NCHW_FORMAT:
#         _, C, H, W = x.shape
#         x_LL = x[:, :, : H // 2, : W // 2]
#         x_LH = x[:, :, H // 2 :, : W // 2]
#         x_HL = x[:, :, : H // 2, W // 2: ]
#         x_HH = x[:, :, H // 2 :, W // 2: ]
#     else:  # data_format == NHWC_FORMAT:
#         _, H, W, C = x.shape
#         x_LL = x[:, : H // 2, : W // 2, :]
#         x_LH = x[:, H // 2 :, : W // 2, :]
#         x_HL = x[:, : H // 2, W // 2 :, :]
#         x_HH = x[:, H // 2 :, W // 2 :, :]
#     return x_LL, x_LH, x_HL, x_HH


# # ----- New vectorized versions -----

# def convert_paddings(pads):
#     # Reverse pads from [dim1, dim2, ... dimN] to [dimN, ..., dim2, dim1] and then flatten
#     new_pads = [pad_value for dim_pad in reversed(pads) for pad_value in dim_pad]
#     return new_pads


# def pos_shift_4d(x, n_shifts, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     # x shape: [N, C, H, W] or [N, H, W, C]
#     # x[i] -> x[i + n], so remove the first n elements and pad on the right side
#     assert not (across_cols and across_rows) and (across_cols or across_rows)
#     if data_format == NCHW_FORMAT:
#         if across_cols:
#             # Inputs have shape NCHW and operation is applied across W axis (cols)
#             paddings = convert_paddings([[0, 0], [0, 0], [0, 0], [0, n_shifts]])
#             padded_x = F.pad(x[:, :, :, n_shifts:], paddings, mode=PAD_MODE)
#         else: # across_rows:
#             # Inputs have shape NCHW and operation is applied across H axis (rows)
#             paddings = convert_paddings([[0,0], [0, 0], [0, n_shifts], [0, 0]])
#             padded_x = F.pad(x[:, :, n_shifts:, :], paddings, mode=PAD_MODE)
#     else:  # data_format == NHWC_FORMAT:
#         if across_cols:
#             # Inputs have shape NHWC and operation is applied across W axis (cols)
#             paddings = convert_paddings([[0, 0], [0, 0], [0, n_shifts], [0, 0]])
#             padded_x = F.pad(x[:, :, n_shifts:, :], paddings, mode=PAD_MODE)
#         else: # across_rows:
#             # Inputs have shape NHWC and operation is applied across H axis (rows)
#             paddings = convert_paddings([[0, 0], [0, n_shifts], [0, 0], [0, 0]])
#             padded_x = F.pad(x[:, n_shifts:, :, :], paddings, mode=PAD_MODE)
#     return padded_x


# def neg_shift_4d(x, n_shifts, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     # x shape: [N, C, H, W] or [N, H, W, C]
#     # x[i] -> x[i - n], so remove the last n elements and pad on the left side
#     assert not(across_cols and across_rows) and (across_cols or across_rows)
#     if data_format == NCHW_FORMAT:
#         if across_cols:
#             # Inputs have shape NCHW and operation is applied across W axis (cols)
#             paddings = convert_paddings([[0, 0], [0, 0], [0, 0], [n_shifts, 0]])
#             padded_x = F.pad(x[:, :, :, :-n_shifts], paddings, mode=PAD_MODE)
#         else: # across_rows:
#             # Inputs have shape NCHW and operation is applied across H axis (rows)
#             paddings = convert_paddings([[0, 0], [0, 0], [n_shifts, 0], [0, 0]])
#             padded_x = F.pad(x[:, :, :-n_shifts, :], paddings, mode=PAD_MODE)
#     else:  # data_format == NHWC_FORMAT:
#         if across_cols:
#             # Inputs have shape NHWC and operation is applied across W axis (cols)
#             paddings = convert_paddings([[0, 0], [0, 0], [n_shifts, 0], [0, 0]])
#             padded_x = F.pad(x[:, :, :-n_shifts, :], paddings, mode=PAD_MODE)
#         else: # across_rows:
#             # Inputs have shape NHWC and operation is applied across H axis (rows)
#             paddings = convert_paddings([[0, 0], [n_shifts, 0], [0, 0], [0, 0]])
#             padded_x = F.pad(x[:, :-n_shifts, :, :], paddings, mode=PAD_MODE)
#     return padded_x











# a1 = -1.58613432
# a2 = -0.05298011854
# a3 = 0.8829110762
# a4 = 0.4435068522
# k = 1.149604398
# d1 = copy.deepcopy(a1) # step 1 for [i]
# d2 = copy.deepcopy(d1) # step 1 for [i+1]
# d3 = copy.deepcopy(a3) # step 2 for [i]
# d4 = copy.deepcopy(d3) # step 2 for [i+1]
# c1 = copy.deepcopy(a2) # step 1 for [i-1]
# c2 = copy.deepcopy(c1) # step 1 for [i]
# c3 = copy.deepcopy(a4) # step 2 for [i-1]
# c4 = copy.deepcopy(c3) # step 2 for [i]
# DEFAULT_KERNEL = [c1, c2, c3, c4, d1, d2, d3, d4, k]
# CDF_97_KERNEL = DEFAULT_KERNEL


# # ----- New vectorized versions -----

# def fast_cdf97_1d_op(x, kernel, scale_coeffs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     # x shape: [N, C, H, W] or [N, H, W, C]
#     assert not(across_cols and across_rows) and (across_cols or across_rows), \
#         f'CDF-9/7 1d op: across_cols = {across_cols}, across_rows = {across_rows}'
#     common_kwargs = {
#         'across_cols': across_cols,
#         'across_rows': across_rows,
#         'data_format': data_format
#     }
#     # Split coeffs
#     x_ev_0, x_od_0 = prepare_coeffs_for_1d_op(x, **common_kwargs)
#     # o - odd, e - even
#     c1, c2, c3, c4, d1, d2, d3, d4, k =\
#         kernel[0], kernel[1], kernel[2], kernel[3], \
#             kernel[4], kernel[5], kernel[6], kernel[7], kernel[8]
#     x_od_1 = x_od_0 + (
#         a1 * x_ev_0 +
#         a1 * pos_shift_4d(x_ev_0, n_shifts=1, **common_kwargs)
#     )
#     x_ev_1 = x_ev_0 + (
#         a2 * neg_shift_4d(x_od_1, n_shifts=1, **common_kwargs) +
#         a2 * x_od_1
#     )
#     x_od_2 = x_od_1 + (
#         a3 * x_ev_1 +
#         a3 * pos_shift_4d(x_ev_1, n_shifts=1, **common_kwargs)
#     )
#     x_ev_2 = x_ev_1 + (
#         a4 * neg_shift_4d(x_od_2, n_shifts=1, **common_kwargs) +
#         a4 * x_od_2
#     )
#     # Normalization
#     if scale_coeffs:
#         x_ev_3 = k * x_ev_2 # s
#         # x_od_3 = (k - 1.) * x_od_2 # d
#         x_od_3 = (1. / k) * x_od_2 # d, use normalization consistent with other families
#     else:
#         x_ev_3, x_od_3 = x_ev_2, x_od_2 # s, d
#     # Join coeffs
#     x = join_coeffs_after_1d_op((x_ev_3, x_od_3), **common_kwargs)
#     return x


# def fast_inv_cdf97_1d_op(x_coefs, kernel, scale_coeffs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
#     # x shape: [N, C, H, W] or [N, H, W, C]
#     assert not(across_cols and across_rows) and (across_cols or across_rows), \
#         f'Inverse CDF-9/7 1d op: across_cols = {across_cols}, across_rows = {across_rows}'
#     common_kwargs = {
#         'across_cols': across_cols,
#         'across_rows': across_rows,
#         'data_format': data_format
#     }
#     # x_coefs: s, d
#     s, d = prepare_coeffs_for_inv_1d_op(x_coefs, **common_kwargs)
#     # o - odd, e - even
#     c1, c2, c3, c4, d1, d2, d3, d4, k =\
#         kernel[0], kernel[1], kernel[2], kernel[3], \
#             kernel[4], kernel[5], kernel[6], kernel[7], kernel[8]
#     if scale_coeffs:
#         x_ev_2 = (1. / k) * s
#         # x_od_2 = (1. / (k - 1.)) * d
#         x_od_2 = k * d # use normalization consistent with other families
#     else:
#         x_ev_2, x_od_2 = s, d
#     x_ev_1 = x_ev_2 - (
#         a4 * neg_shift_4d(x_od_2, n_shifts=1, **common_kwargs) +
#         a4 * x_od_2
#     )
#     x_od_1 = x_od_2 - (
#         a3 * x_ev_1 +
#         a3 * pos_shift_4d(x_ev_1, n_shifts=1, **common_kwargs)
#     )
#     x_ev_0 = x_ev_1 - (
#         a2 * neg_shift_4d(x_od_1, n_shifts=1, **common_kwargs) +
#         a2 * x_od_1
#     )
#     x_od_0 = x_od_1 - (
#         a1 * x_ev_0 +
#         a1 * pos_shift_4d(x_ev_0, n_shifts=1, **common_kwargs)
#     )
#     # Join coeffs
#     x = join_coeffs_after_inv_1d_op((x_ev_0, x_od_0), src_shape=x_coefs.shape, **common_kwargs)
#     return x


# #@tf.function(jit_compile=True)
# def fast_cdf97_2d_op(x, kernel, scale_1d_coeffs, scale_2d_coeffs, coeffs_scales_2d, data_format=DEFAULT_DATA_FORMAT):
#     # 1. Apply across rows
#     x = fast_cdf97_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=False, across_rows=True, data_format=data_format)
#     # 2. Apply across cols
#     x = fast_cdf97_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=True, across_rows=False, data_format=data_format)
#     # 3. Rearrange images from spatial into channels
#     x_LL, x_LH, x_HL, x_HH = extract_coeffs_from_spatial(x, data_format=data_format)
#     if scale_2d_coeffs:
#         coeffs_scales = coeffs_scales_2d
#         x_LL *= coeffs_scales[0]
#         x_LH *= coeffs_scales[1]
#         x_HL *= coeffs_scales[2]
#         x_HH *= coeffs_scales[3]
#     x_output = merge_coeffs_into_channels([x_LL, x_LH, x_HL, x_HH], data_format=data_format)
#     return x_output


# def fast_inv_cdf97_2d_op(x, kernel, scale_1d_coeffs, scale_2d_coeffs, coeffs_scales_2d, data_format=DEFAULT_DATA_FORMAT):
#     # x_LL, x_LH, x_HL, x_HH = x_coeffs
#     # 1. Rearrange images from channels into spatial
#     x_LL, x_LH, x_HL, x_HH = extract_coeffs_from_channels(x, data_format=data_format)
#     if scale_2d_coeffs:
#         coeffs_scales = 1. / coeffs_scales_2d
#         x_LL *= coeffs_scales[0]
#         x_LH *= coeffs_scales[1]
#         x_HL *= coeffs_scales[2]
#         x_HH *= coeffs_scales[3]
#     x = merge_coeffs_into_spatial([x_LL, x_LH, x_HL, x_HH], data_format=data_format)
#     # 2. Apply inverse transform across cols
#     x = fast_inv_cdf97_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=True, across_rows=False, data_format=data_format)
#     # 3. Apply inverse transform across rows
#     x = fast_inv_cdf97_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=False, across_rows=True, data_format=data_format)
#     return x






# class CDFLifting2DVectorized(nn.Module):
#     def __init__(self, levels=1, data_format='NCHW'):
#         super(CDFLifting2DVectorized, self).__init__()
#         self.levels = levels
#         self.data_format = data_format
#         self.kernel = DEFAULT_KERNEL
#         self.coeffs_scales_2d = COEFFS_SCALES_2D

#     def forward(self, x):
#         for _ in range(self.levels):
#             x = fast_cdf97_2d_op(x, self.kernel, scale_1d_coeffs=True, scale_2d_coeffs=True, coeffs_scales_2d=self.coeffs_scales_2d, data_format=self.data_format)
#         return x

#     def inverse(self, x):
#         for _ in range(self.levels):
#             x = fast_inv_cdf97_2d_op(x, self.kernel, scale_1d_coeffs=True, scale_2d_coeffs=True, coeffs_scales_2d=self.coeffs_scales_2d, data_format=self.data_format)
#         return x

# class WaveletCNN(nn.Module):
#     def __init__(self, in_channels, class_nums, levels=1, base_channels=16, dropout_rate=0.3):
#         super(WaveletCNN, self).__init__()
#         self.levels = levels
#         self.dwt = CDFLifting2DVectorized(levels=levels)
#         self.projections = nn.ModuleList()
#         self.conv_blocks = nn.ModuleList()
#         self.level_channels = []

#         # Adjust the input_channels calculation here
#         current_in_channels = in_channels * (4 ** levels) # Adjusted line

#         for i in range(levels):
#             prev_channels_sum = sum(self.level_channels)
#             input_channels = current_in_channels + prev_channels_sum
#             output_channels = base_channels * (2 ** i)
#             self.level_channels.append(output_channels)
#             self.projections.append(nn.Conv2d(input_channels, output_channels, kernel_size=1))
#             self.conv_blocks.append(nn.Sequential(
#                 nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(output_channels),
#                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                 nn.Dropout(p=dropout_rate)
#             ))
#             current_in_channels = output_channels

#         total_channels = sum(self.level_channels)
#         self.seg_head = nn.Sequential(
#             nn.Conv2d(total_channels, total_channels // 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(total_channels // 2),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.Dropout(p=dropout_rate),
#             nn.Conv2d(total_channels // 2, total_channels // 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(total_channels // 2),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True)
#         )

#         self.upsample_layers = nn.ModuleList()
#         current_channels = total_channels // 2

#         for i in range(levels):
#             out_channels = class_nums
#             self.upsample_layers.append(
#                 nn.ConvTranspose2d(current_channels, out_channels, kernel_size=2, stride=2)
#             )
#             current_channels = out_channels

#         self.downsample = nn.Conv2d(class_nums, class_nums, kernel_size=2, stride=2)

#     def forward(self, x):
#         features = []
#         low_pass = x

#         for i in range(self.levels):
#             low_pass = self.dwt(low_pass)
#             batch_size, channels, height, width = low_pass.shape

#             resized_features = [
#                 F.interpolate(feat, size=(height, width), mode='bilinear', align_corners=False)
#                 for feat in features
#             ]

#             concatenated = torch.cat([low_pass] + resized_features, dim=1)
#             projected = self.projections[i](concatenated)
#             conv_out = self.conv_blocks[i](projected)
#             features.append(conv_out)

#         smallest_height, smallest_width = features[-1].shape[2:]
#         resized_features = [
#             F.interpolate(feat, size=(smallest_height, smallest_width), mode='bilinear', align_corners=False)
#             for feat in features
#         ]

#         out = torch.cat(resized_features, dim=1)
#         out = self.seg_head(out)

#         # Upsample to target size
#         for upsample_layer in self.upsample_layers:
#             out = upsample_layer(out)

#         out = F.interpolate(out, size=(16, 16), mode='bilinear', align_corners=False) # Ensure output matches target size

#         return out

# # Example Usage with summary:
# num_components = 120
# NUM_CLASS = 3

# net = WaveletCNN(in_channels=num_components, class_nums=NUM_CLASS)

# # Input tensor
# input_tensor = torch.randn(1, num_components, PATCH_SIZE, PATCH_SIZE)
# print(f"Input tensor shape: {input_tensor.shape}")

# # Model summary
# summary(
#     net,
#     input_size=(1, num_components, PATCH_SIZE, PATCH_SIZE),
#     col_names=['num_params', 'kernel_size', 'mult_adds', 'input_size', 'output_size'],
#     col_width=18,
#     row_settings=['var_names'],
#     depth=4,
# )