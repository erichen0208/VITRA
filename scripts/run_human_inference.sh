# python scripts/inference_human_prediction.py \
#     --config VITRA-VLA/VITRA-VLA-3B \
#     --image_path ./examples/0002.jpg \
#     --sample_times 4 \
#     --save_state_local \
#     --use_right \
#     --video_path ./example_human_inf.mp4 \
#     --mano_path ./weights/mano \
#     --instruction "Left hand: None. Right hand: Pick up the picture of Michael Jackson." \

# python scripts/inference_human_prediction.py \
#     --config VITRA-VLA/VITRA-VLA-3B \
#     --image_path ./examples/0001.jpg \
#     --sample_times 4 \
#     --use_left \
#     --video_path ./example_human_inf.mp4 \
#     --mano_path ./weights/mano \
#     --instruction "Left hand: Put the trash into the garbage. Right hand: None." \

# python scripts/inference_human_prediction.py \
#     --config VITRA-VLA/VITRA-VLA-3B \
#     --image_path ./examples/0003.png \
#     --sample_times 4 \
#     --use_right \
#     --video_path ./example_human_inf.mp4 \
#     --mano_path ./weights/mano \
#     --instruction "Left hand: None. Right hand: Pick up the metal water cup." \

python scripts/inference_human_prediction.py \
    --config VITRA-VLA/VITRA-VLA-3B \
    --image_path ./examples/custom/full_hand.jpg \
    --sample_times 1 \
    --use_left \
    --video_path ./examples/custom/output.mp4 \
    --mano_path ./weights/mano \
    --instruction "Left hand: Pick up the mug. Right hand: None." \