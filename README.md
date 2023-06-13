# NIQ_store_type_text_classification_with_BERT 
- The repository is a personal backup of my effort in upgrading pipelines and models for the 'Store Advisor' project at NielsenIQ. :tada:
- Data shall not be provided due to the Non Disclosure Agreement.
- Any suggestion will be welcomed.
- Finished:
- [x] data cleaning by 0321
- [x] modeling by 0521
- [x] pipeline by 0611
- to do:
- [ ] build config.py to control parameters by 0701 (I was too young too naive 😹)
# Functions :rocket:
The project code can be reused for other text classification tasks with the following functions:
- Automatically download models and tokenizers from Huggingface by simply inputing their model name.
- Automatically training, testing, logging, and checking point for any dfs with 'text' and 'label' columns.
- Designed pipelines for incremental training from check points.

# Usage
you can...
- Prepare cleaned train valid test datasets with columns such as 'text' and 'label'
- Look for pretrained language models at Huggingface that interest you
- Copy the model name and paste it the the 'model_name' variable as in main.py
- :wrench: Change relevent parameters like learning rate, batch size, etc... as stated in main.py 
- :wrench: Try different tricks and parameters, such as focal loss, class balancing, which I have written in the tool folder
- Once obtained the best model, you can also do incremental training on additional data and I have prepared the code for you 


