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
- classes.txt

```text
zero
one
two
three
four
five
six
seven
eight
nine
```

- label_map.txt

```text
item {
    name: "zero",
    id: 1,
    display_name: "zero"
}
item {
    name: "one",
    id: 2,
    display_name: "one"
}
item {
    name: "two",
    id: 3,
    display_name: "two"
}
item {
    name: "three",
    id: 4,
    display_name: "three"
}
item {
    name: "four",
    id: 5,
    display_name: "four"
}
item {
    name: "five",
    id: 6,
    display_name: "five"
}
item {
    name: "six",
    id: 7,
    display_name: "six"
}
item {
    name: "seven",
    id: 8,
    display_name: "seven"
}
item {
    name: "eight",
    id: 9,
    display_name: "eight"
}
item {
    name: "nine",
    id: 10,
    display_name: "nine"
}
```

- jpg and xml files

![vaik-mnist-detection-dataset-output](https://user-images.githubusercontent.com/116471878/198033194-8282c30a-bce0-4634-86bc-a85c83ccd1a8.png)
