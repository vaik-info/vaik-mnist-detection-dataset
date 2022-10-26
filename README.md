# vaik-mnist-detection-dataset

Create Pascal VOC formatted MNIST detection dataset

## Example

![vaik-mnist-detection-dataset](https://user-images.githubusercontent.com/116471878/198029678-ce667f2f-f1eb-44c0-b737-33c94ff47125.png)

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

## Output

![vaik-mnist-detection-dataset-output](https://user-images.githubusercontent.com/116471878/198033194-8282c30a-bce0-4634-86bc-a85c83ccd1a8.png)
