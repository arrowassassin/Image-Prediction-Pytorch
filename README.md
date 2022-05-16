# Image-Prediction-Pytorch

To test out the code, the folloing rules need to be followed - 

1. The CIFAR-10 data has not been downloaded. The dataset needs to be placed in the main folder with the name "cifar-10-batches-py".
   The final path should somewhat look like this - "C:/Users/Piyush Nayak/OneDrive/Documents/TAMU assignments/DL/Final Project/cifar-10-batches-py".
   Final Project here is the main folder, which contains the subfolders like code and saved_models.

2. The following configurations are needed to be understood to customisably run the model - 

"--datadir", default = "/content/drive/MyDrive/Project-mobilenet/Project/" :- Pass the location of the main folder here which contains all of the sub folders
"--oper", default = "train" :- Pass the operation whiich is needed to be done, namely train, test and predict
'--lr', default=0.1 :- You can customise the learning rate here
"--save_interval", default=10 :- The model checkpoint intervals can be saved here
"--batch_size", default = 256 :- You can customise the batch size using this
"--weight_decay", default=1e-4 :- The weight decay can be changed using this
"--modeldir", default='model_v11' :- The name of the folder where the checkpoints will be stores. In this case a new folder called model_v11 will be created
"--maxepochs", default=200 :- Total number of epochs to be performed
"--testepoch", default=190 :- The model checkpoint to be used to perform the test operation on the CIFAR-10 test dataset

3. To perform train operation, use the follwoing line of code -

!python main.py --datadir "C:/Users/XYZ/Project" --oper "train" --maxepochs 200 --modeldir "model_v11" --lr 0.01 

Incase of more customisations, you can add --params(ex - batch_size, weight_decay, etc) to the above mentioned line of code accordingly.

4. To perform test operation, use the following line of code - 

!python main.py --datadir "C:/Users/XYZ/Project" --oper "test" --testepoch 190 --modeldir "model_v11" 

5. To perform predict operation, use the following line of code - 

!python main.py --datadir "C:/Users/XYZ/Project" --oper "predict" --testepoch 190 --modeldir "model_v11" 

Kindly, keep the private_images.npy file in the same main folder as the other sub-folders. 
