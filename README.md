# NIQ Store Type Text Classification with BERT

This repository serves as a personal backup of my efforts in upgrading pipelines and models for the 'Store Advisor' project at NielsenIQ. :tada:

**Note:** Due to the Non-Disclosure Agreement, data cannot be provided in this repository.

Any suggestions are welcomed.

## Project Status
- [x] Data cleaning (Completed by 0321)
- [x] Modeling (Completed by 0521)
- [x] Pipeline development (Completed by 0611)

To-do:
- [ ] Build `config.py` to control parameters (I was too young, too naive 😹) (Target completion date: 0701)
- [ ] Build [MIT License](LICENSE) when everything is ready (Target completion date: 0711)
## Functions :rocket:

The project code can be reused for other text classification tasks and offers the following functions:

- Automatic download of models and tokenizers from Huggingface by providing the model name.
- Automated training, testing, logging, and checkpointing for any DataFrames with 'text' and 'label' columns.
- Designed pipelines for incremental training from checkpoints.

## Usage

To use this project, follow these steps:

1. Prepare cleaned train, valid, and test datasets with 'text' and 'label' columns.
2. Look for pretrained language models on Huggingface that interest you.
3. Copy the model name and paste it into the `model_name` variable in `main.py`.
4. :wrench: Modify relevant parameters such as learning rate, batch size, etc., as specified in `main.py`.
5. :wrench: Experiment with different tricks and parameters, such as focal loss and class balancing, which are provided in the `tool` folder.
6. Once you obtain the best model, you can perform incremental training on additional data. The code is already prepared for you.

Please refer to the documentation within the code for more detailed usage instructions and examples.

## Contributions

Contributions to this project are welcome! If you have any suggestions, ideas, or improvements, please feel free to open an issue or submit a pull request.

## License

This project will be licensed under the [MIT License](LICENSE) once finalized
