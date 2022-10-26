# vaik-mnist-detection-dataset

Create Pascal VOC formatted MNIST detection dataset

## Usage

```shell
pip install -r requirements.txt
python main.py --output_dir_path ~/.vaik-mnist-detection-dataset \
                --train_sample_num 10000 \
                --valid_sample_num 100 \
                --image_max_size 768 \
                --image_min_size 256 \
                --char_max_size 128 \
                --char_min_size 64 \
                --char_max_num 6 \
                --char_min_num 2
```