# from ImageUtils import parse_record
from DataLoader import load_data, train_valid_split, load_testing_images
from Model import MyModel
import numpy as np
from Configure import configure
import os


def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    
    ### YOUR CODE HERE

    x_train, y_train, x_test, y_test = load_data(os.path.join(config.datadir, "cifar-10-batches-py/"))
    x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

    model = MyModel(config)

    if config.oper == "train":
      model.train(x_train, y_train, config.maxepochs)

    elif config.oper == "test":
      model.evaluate(x_test, y_test, [config.testepoch])

    elif config.oper == "predict":
      data_dir_1 = config.datadir + "private_test_images_v3.npy"
      x_test_priv = load_testing_images(data_dir_1)
      answers = model.predict_prob(x_test_priv, [config.testepoch])

      np.save(config.datadir + "private_answers_" + config.modeldir + ".npy", answers)


    ### YOUR CODE HERE
    # First step: use the train_new set and the valid set to choose hyperparameters.
    # model.train(x_train_new, y_train_new, 200)
    # model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])
    # model.test_or_validate(x_valid, y_valid, [10])


    # Second step: with hyperparameters determined in the first run, re-train
    # your model on the original train set.
    

    # Third step: after re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report.
    # for x in range(10, 200, 10):
    #   model.test_or_validate(x_test, y_test, [x])
    # model.test_or_validate(x_test, y_test, [10])

    ### END CODE HERE


    # Saving in the predicted set

    # data_dir_1 = "/content/drive/MyDrive/Project-mobilenet/Project/private_test_images_v3.npy"
    # x_test_priv = load_testing_images(data_dir_1)
    # # for x in range(10, 210, 10):
    # #   answers = model.test_private_data(x_test_priv, [x])
    # #   zz = [np.argmax(x) for x in answers]
    # #   print("Sets of answers - ", {x:zz.count(x) for x in set(zz)})
    # answers = model.test_private_data(x_test_priv, [190])

    # np.save("/content/drive/MyDrive/Project-mobilenet/Project/private_answers_model_v11.npy", answers)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)