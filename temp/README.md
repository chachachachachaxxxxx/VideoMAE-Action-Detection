run_class_finetuning_single 这个文件是我用来魔改产生一个单卡训练的结果吧

示例：

1) 通过 YAML 配置训练（命令行可覆写 YAML 中的参数）：

```bash
python temp/run_class_finetuning_single.py --config temp/configs/ava_single.yaml \
  --output_dir temp/outputs --log_dir temp/tb
```

2) 直接命令行训练：

```bash
python temp/run_class_finetuning_single.py \
  --data_set ava --batch_size 8 --epochs 30 \
  --output_dir temp/outputs --log_dir temp/tb
```