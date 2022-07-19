import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns


class ModelResults():
    def __init__(self, model, result_dict, output_path, testset= None):

        self.model = model
        self.result_dict = result_dict
        self.output_path = output_path

        if testset:
            self.testset = testset

        self.epochs = result_dict['epochs']
        self.loss = result_dict['loss']
        self.acc = result_dict['acc']
        self.lr = result_dict['lr']

    def save_results(self):
        
        plt.plot(self.epochs, self.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.output_path + 'loss_vs_epochs.png')
        plt.close()

        plt.plot(self.epochs, self.lr)
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.savefig(self.output_path + 'loss_vs_lr.png')
        plt.close()

        if self.acc:
            plt.plot(self.epochs, self.acc)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.savefig(self.output_path + 'acc_vs_epochs.png')
            plt.close()

