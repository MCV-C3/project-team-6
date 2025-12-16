python3 -m experiments.experiment_simple_model --epochs 500 --gpu-id 0 > experiment_simple_model.log
python3 -m experiments.experiment_widths --width 32 --epochs 500 --gpu-id 0 > experiment_widths_32.log
python3 -m experiments.experiment_widths --width 64 --epochs 500 --gpu-id 0 > experiment_widths_64.log
python3 -m experiments.experiment_widths --width 128 --epochs 500 --gpu-id 0 > experiment_widths_128.log
python3 -m experiments.experiment_widths --width 256 --epochs 500 --gpu-id 0 > experiment_widths_256.log
python3 -m experiments.experiment_widths --width 512 --epochs 500 --gpu-id 0 > experiment_widths_512.log
python3 -m experiments.experiment_widths --width 1024 --epochs 500 --gpu-id 0 > experiment_widths_1024.log
python3 -m experiments.experiment_widths --width 2048 --epochs 500 --gpu-id 0 > experiment_widths_2048.log





python3 -m experiments.experiment_depths --depth 1 --epochs 500 --gpu-id 0 > experiment_depths_1.log
python3 -m experiments.experiment_depths --depth 2 --epochs 500 --gpu-id 0 > experiment_depths_2.log
python3 -m experiments.experiment_depths --depth 3 --epochs 500 --gpu-id 0 > experiment_depths_3.log
python3 -m experiments.experiment_depths --depth 4 --epochs 500 --gpu-id 0 > experiment_depths_4.log
python3 -m experiments.experiment_depths --depth 5 --epochs 500 --gpu-id 0 > experiment_depths_5.log
python3 -m experiments.experiment_depths --depth 6 --epochs 500 --gpu-id 0 > experiment_depths_6.log




python3 -m experiments.experiment_image_sizes --depth 5 --width 300 --imsize 16 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 5 --width 300 --imsize 32 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 5 --width 300 --imsize 64 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 5 --width 300 --imsize 128 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 5 --width 300 --imsize 224 --epochs 500 --gpu-id 0

python3 -m experiments.experiment_image_sizes --depth 2 --width 512 --imsize 16 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 2 --width 512 --imsize 32 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 2 --width 512 --imsize 64 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 2 --width 512 --imsize 128 --epochs 500 --gpu-id 0
python3 -m experiments.experiment_image_sizes --depth 2 --width 512 --imsize 224 --epochs 500 --gpu-id 0

python3 -m experiments.experiment_patch_based --merge-strategy mean --patch-size 4 --epochs 500
python3 -m experiments.experiment_patch_based --merge-strategy mean --patch-size 8 --epochs 500
python3 -m experiments.experiment_patch_based --merge-strategy mean --patch-size 16 --epochs 500
python3 -m experiments.experiment_patch_based --merge-strategy mean --patch-size 32 --epochs 500

python3 -m experiments.experiment_pyramidal_default --epochs 1000
python3 -m experiments.experiment_pyramidal_fine_to_coarse --epochs 1000

python3 -m experiments.experiment_patch_based_last_layer_width --patch-size 32 --last-layer-width 32 --epochs 500 --gpu-id 0 > experiment_last_layer_width_32.log
python3 -m experiments.experiment_patch_based_last_layer_width --patch-size 32 --last-layer-width 64 --epochs 500 --gpu-id 0 > experiment_last_layer_width_64.log
python3 -m experiments.experiment_patch_based_last_layer_width --patch-size 32 --last-layer-width 128 --epochs 500 --gpu-id 0 > experiment_last_layer_width_128.log
python3 -m experiments.experiment_patch_based_last_layer_width --patch-size 32 --last-layer-width 256 --epochs 500 --gpu-id 0 > experiment_last_layer_width_256.log
python3 -m experiments.experiment_patch_based_last_layer_width --patch-size 32 --last-layer-width 512 --epochs 500 --gpu-id 0 > experiment_last_layer_width_512.log
python3 -m experiments.experiment_patch_based_last_layer_width --patch-size 32 --last-layer-width 1024 --epochs 500 --gpu-id 0 > experiment_last_layer_width_1024.log


# Without LDA
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw32_d2/best_test_loss.pt --patch-size 32 --last-layer-width 32 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw32_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw64_d2/best_test_loss.pt --patch-size 32 --last-layer-width 64 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw64_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw128_d2/best_test_loss.pt --patch-size 32 --last-layer-width 128 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw128_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw256_d2/best_test_loss.pt --patch-size 32 --last-layer-width 256 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw256_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw512_d2/best_test_loss.pt --patch-size 32 --last-layer-width 512 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw512_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw1024_d2/best_test_loss.pt --patch-size 32 --last-layer-width 1024 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw1024_no_lda.log

# With LDA (8 components)
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw32_d2/best_test_loss.pt --patch-size 32 --last-layer-width 32 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw32_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw64_d2/best_test_loss.pt --patch-size 32 --last-layer-width 64 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw64_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw128_d2/best_test_loss.pt --patch-size 32 --last-layer-width 128 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw128_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw256_d2/best_test_loss.pt --patch-size 32 --last-layer-width 256 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw256_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw512_d2/best_test_loss.pt --patch-size 32 --last-layer-width 512 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw512_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw1024_d2/best_test_loss.pt --patch-size 32 --last-layer-width 1024 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw1024_lda8.log


# Without LDA
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw32_d2/best_test_loss.pt --patch-size 32 --last-layer-width 32 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw32_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw64_d2/best_test_loss.pt --patch-size 32 --last-layer-width 64 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw64_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw32_d2/best_test_loss.pt --patch-size 32 --last-layer-width 32 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw32_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw64_d2/best_test_loss.pt --patch-size 32 --last-layer-width 64 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw64_lda8.log


python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw128_d2/best_test_loss.pt --patch-size 32 --last-layer-width 128 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw128_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw256_d2/best_test_loss.pt --patch-size 32 --last-layer-width 256 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw256_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw128_d2/best_test_loss.pt --patch-size 32 --last-layer-width 128 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw128_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw256_d2/best_test_loss.pt --patch-size 32 --last-layer-width 256 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw256_lda8.log


python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw512_d2/best_test_loss.pt --patch-size 32 --last-layer-width 512 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw512_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw1024_d2/best_test_loss.pt --patch-size 32 --last-layer-width 1024 --depth 2 --num-words 512 --gpu-id 0 > bovw_lw1024_no_lda.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw512_d2/best_test_loss.pt --patch-size 32 --last-layer-width 512 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw512_lda8.log
python3 -m experiments.experiment_bovw --checkpoint-path trained_models/last_layer_width_ps32_s32_mean_lw1024_d2/best_test_loss.pt --patch-size 32 --last-layer-width 1024 --depth 2 --num-words 512 --lda-components 8 --gpu-id 0 > bovw_lw1024_lda8.log

