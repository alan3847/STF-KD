#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  -d mmnist
  --config_file ./configs/mmnist/ConvLSTM.py
  --student_method convlstm
  --teacher_method simvp
  --teacher_config_file ./configs/mmnist/simvp/SimVP_gSTA.py
  --teacher_ckpt ./work_dirs/mmnist_simvp_s_gsta_one_ep200/mmnist_simvp_s_gsta_one_ep200.pth
  --student_hint_layer model.cell_list.3
  --teacher_hint_output_index 0
  --student_hint_output_index 0
  --hint_teacher_channels 64
  --hint_student_channels 128
  --kd_weight 0.5
  --freq_alpha 1.0
  --freq_cutoff 0.25
  --freq_loss_type magnitude
  --freq_norm rms
  --freq_eps 1e-6
  --freq_log_mag False
  --output_kd True
  --data_root ./data
)

run_fakd() {
  local idx="$1"
  local name="$2"
  local teacher_hint_layer="$3"
  local hint_reduce="$4"
  local fakd_weight="$5"

  echo "========== [$idx/2] $name =========="

  PYTHONDONTWRITEBYTECODE=1 python -m tools.train_fakd_unified \
    "${COMMON_ARGS[@]}" \
    --teacher_hint_layer "$teacher_hint_layer" \
    --hint_reduce "$hint_reduce" \
    --fakd_weight "$fakd_weight" \
    --ex_name "$name"
}

run_fakd 1 fakd_enc_last_w001  model.enc last 0.01
run_fakd 2 fakd_enc_last_w0005 model.enc last 0.005

echo "========== All FAKD experiments finished =========="
