# 1.Extract Embedding
"python camera_demo.py -c 0 -t 0.5 -m tf"
# 1.Extract Embedding
 - Download model [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn) and paste to folder
same folder with file extract.py.
 - Create folder of images,example :
    - create :
      - your_folder/:
        - person1/:
        - person2/:
        - ...
    - person1 : folder of person1 images.
  - Enter command line :```python extract.py``` extract embedding of images.
    - output : embeddings folder .After you extract,each person has a file .pickle, ex : person1.pickle.
    
# 2.Training and Testing with SCV model
- **Training**
  - Enter command line :```python svc_model.py -m train -e <embds-path> -n <max_num_image> -model <model-path>```.
    - Use SVC model training classification of embddings (after extract embdding).
      - embds-path : folder of embeddings.
      - max_num_image : max num embddings for each one.
      - model-path : output of svc model.
- **Testing**
  - Enter command line :```python svc_model.py -m test -model <model-path>``` extract embedding of images.
    - Use SVC model training classification of embddings (after extract embdding).
      - model-path : folder of svc model.
    - After running , system ask you to enter filename of image.
    - Output in labels.txt
