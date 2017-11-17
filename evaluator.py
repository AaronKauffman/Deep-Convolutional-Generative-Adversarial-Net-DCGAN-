import tensorflow as tf
import tensorflow.contrib.slim as slim
import cPickle
import numpy as np
import math
import os
import scipy.misc


class Evaluator(object):
	
	def __init__(self, model, batch_size=128, update_ratio=2, train_iter=10, cifar_dir='/home/robot/Dataset/cifar10_data/', log_dir='/home/robot/Experiments/DCGAN/logs/', save_model_dir='/home/robot/Experiments/DCGAN/trained_models/', save_sample_dir='/home/robot/Experiments/DCGAN/sampled_images/', test_model='/home/robot/Experiments/DCGAN/trained_models/dcgan-10'):
		
		self.model = model
		self.batch_size = batch_size
		self.update_ratio = update_ratio
		self.train_iter = train_iter
		self.cifar_dir = str(cifar_dir)
		self.log_dir = str(log_dir)
		self.save_model_dir = str(save_model_dir)
		self.save_sample_dir = str(save_sample_dir)
		self.test_model = str(test_model)
		
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		
	def load_cifar(self, source='train'):
		if source=='train':
			# load images for training
			images = []
			for i in xrange(1, 6):
				with open(self.cifar_dir+'data_batch_'+str(i), 'rb') as t_cifar:
					t_cifar_dict = cPickle.load(t_cifar)
					t_cifar_img = t_cifar_dict['data']
					images.append(t_cifar_img)
	
			images = np.array(images)
		else:
			# load images for evaluation
			with open(self.cifar_dir+'test_batch', 'rb') as v_cifar:
				v_cifar_dict = cPickle.load(v_cifar)
				v_cifar_img = v_cifar_dict['data']
		
			images = np.array(v_cifar_img)
			
		images = np.reshape(images, [-1, 3, 32, 32])
		images = np.transpose(images, [0, 2, 3, 1])
		images = images / 127.5 - 1	
		return images
		
	def train(self):
		train_images = self.load_cifar(source='train')
		model = self.model
		model.build_model()
		
		# make directories
		if tf.gfile.Exists(self.log_dir)==0:
			tf.gfile.MakeDirs(self.log_dir)
		if tf.gfile.Exists(self.save_model_dir)==0:
			tf.gfile.MakeDirs(self.save_model_dir)
		
		with tf.Session(config=self.config) as sess:
			# initialize
			tf.global_variables_initializer().run()
			
			summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
			saver = tf.train.Saver()
			
			print('start training...')
			batches = int(math.floor(train_images.shape[0] / self.batch_size))
			
			for step in xrange(self.train_iter):
			
				print('Epoch [%d] \r\ntraining...' %(step+1))
				
				# perform training
				for batch in xrange(batches):
					# train generator
					for update in xrange(self.update_ratio):
						# print('train generator on batch %d' %(batch+1))
						fx_batch = np.random.uniform(-1, 1, (self.batch_size, 1, 1, model.fx_dim)) 
						feed_dict = {model.fx: fx_batch}
						sess.run(model.g_train_op, feed_dict)
					
					# train discriminator
					# print('train discriminator on batch %d' %(batch+1))	
					img_batch = train_images[batch*self.batch_size:(batch+1)*self.batch_size, :]
					fx_batch = np.random.uniform(-1, 1, (self.batch_size, 1, 1, model.fx_dim))
					feed_dict = {model.images: img_batch, model.fx: fx_batch}
					sess.run(model.d_train_op, feed_dict)
				
				# train on rest data
				for update in xrange(self.update_ratio):
					fx_batch = np.random.uniform(-1, 1, ((train_images.shape[0]-batches*self.batch_size), 1, 1, model.fx_dim))
					tf.random_uniform([(train_images.shape[0]-batches*self.batch_size), model.fx_dim], -1, 1)
					feed_dict = {model.fx: fx_batch}
					sess.run(model.g_train_op, feed_dict)
				
				img_batch = train_images[batches*self.batch_size:train_images.shape[0], :]
				fx_batch = np.random.uniform(-1, 1, ((train_images.shape[0]-batches*self.batch_size), 1, 1, model.fx_dim))
				feed_dict = {model.images: img_batch, model.fx: fx_batch}
				sess.run(model.d_train_op, feed_dict)
				
				summary, d_loss, g_loss = sess.run([model.summary, model.d_loss, model.g_loss], feed_dict)
				summary_writer.add_summary(summary, step)
				print('Epoch [%d] ->train_result \r\ndiscriminator_loss: %.6f \r\ngenerator_loss: %.6f \r\n' %(step+1, d_loss, g_loss))
				
				saver.save(sess, os.path.join(self.save_model_dir, 'dcgan'), global_step=step+1)
				print('dcgan-%d saved' %(step+1))
				
			
	def eval(self):
		model = self.model
		model.build_model()
			
		# make directory
		if tf.gfile.Exists(self.save_sample_dir)==0:
			tf.gfile.MakeDirs(self.save_sample_dir)
			
		with tf.Session(config=self.config) as sess:
			# load trained parameters
			print('loading trained generator...')
			saver = tf.train.Saver()
			saver.restore(sess, self.test_model)
				
			# perform sampling
			print('start sampling...')
			fx_batch = np.random.uniform(-1, 1, (self.batch_size, 1, 1, model.fx_dim))
			feed_dict = {model.fx: fx_batch}
			sampled_images, pass_rate = sess.run([model.sampled_images, model.pass_rate], feed_dict)
			for img in xrange(self.batch_size):
			 	path = os.path.join(self.save_sample_dir, 'sample_%d.png' %(img))
			 	scipy.misc.imsave(path, sampled_images[img,:])
			# image_1 = sampled_images[0:64, :]
			# image_1 = np.reshape(image_1, [8, 8, 32, 32, 3])
			# image_1 = np.reshape(image_1, [8, 256, 32, 3])
			# image_1 = np.transpose(image_1, [0, 2, 1, 3])
			# image_1 = np.reshape(image_1, [256, 256, 3])
			# scipy.misc.imsave('/home/robot/Experiments/DCGAN/sample_1.png', image_1)
			# image_2 = sampled_images[64:128, :]
			# image_2 = np.reshape(image_2, [8, 8, 32, 32, 3])
			# image_2 = np.reshape(image_2, [8, 256, 32, 3])
			# image_2 = np.transpose(image_2, [0, 2, 1, 3])
			# image_2 = np.reshape(image_2, [256, 256, 3])
			# scipy.misc.imsave('/home/robot/Experiments/DCGAN/sample_2.png', image_2)
			print ('sample images saved in %s' %(self.save_sample_dir))
			print ('Passing Rate for these samples is %f given by trained discriminator' %(pass_rate))
