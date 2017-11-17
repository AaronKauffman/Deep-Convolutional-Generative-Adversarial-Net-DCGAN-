from model import DCGAN
from evaluator import Evaluator
import argparse

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode')
	
	args = parser.parse_args()
	
	model = DCGAN(mode=args.mode, learning_rate=0.0002, fx_dim=100, beta=0.5)
	evaluator = Evaluator(model, batch_size=128, update_ratio=2, train_iter=20, cifar_dir='/home/robot/Dataset/cifar10_data/', log_dir='/home/robot/Experiments/DCGAN/logs/', save_model_dir='/home/robot/Experiments/DCGAN/trained_models/', save_sample_dir='/home/robot/Experiments/DCGAN/sampled_images/', test_model='/home/robot/Experiments/DCGAN/trained_models/dcgan-20')
	
	if args.mode == 'train':
		evaluator.train()
	else:
		evaluator.eval()
