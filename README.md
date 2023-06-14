# Auto Text Classification with BERT
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
This repository serves as a personal backup of my efforts in upgrading pipelines and models for the 'Store Advisor' project at NielsenIQ. :tada:

**Note:** Due to the Non-Disclosure Agreement, data cannot be provided in this repository.

## Project Status
- [x] Data cleaning (Completed by 0321)
- [x] Modeling (Completed by 0521)
- [x] Pipeline development (Completed by 0611)

To-do:
- [ ] Finalize documentation and in-line doc str (Target completion date: 0618)
- [ ] Build `config.py` to control parameters (I was too young, too naive 😹) (Target completion date: 0701)

## Functions :rocket:

The project code can be reused for other text classification tasks and offers the following functions:

- Automatic downloading of models and tokenizers from Huggingface by simply providing the model name.
- Automated training, testing, logging, and checkpointing for any DataFrames with 'text' and 'label' columns.
- Designed pipelines for incremental training using checkpoints.

## Usage :computer:

To use this project, follow these steps:

1. Prepare cleaned training, validation, and testing datasets with columns such as 'text' and 'label'.
2. Explore the available pretrained language models on Huggingface that interest you.
3. Copy the model name and paste it into the `model_name` variable in `main.py`.
4. :wrench: Adjust relevant parameters such as learning rate, batch size, etc., as specified in `main.py`.
5. :wrench: Try out different techniques and parameters, such as focal loss and class balancing, which are provided in the `tools` folder.
6. Once you have obtained the best model, you can also perform incremental training on additional data, and the necessary code is already provided for you.

Please refer to the documentation within the code for more detailed usage instructions and examples.

## Contributions

Contributions to this project are welcome! If you have any suggestions, ideas, or improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
