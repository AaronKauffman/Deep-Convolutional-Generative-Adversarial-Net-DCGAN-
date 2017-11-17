import tensorflow as tf
import tensorflow.contrib.slim as slim


class DCGAN(object):
	"Deep Convolutional Generative Adversarial Net"
	def __init__(self, mode='train', learning_rate=0.0002, fx_dim=100, beta=0.5, gamma1=0.05, gamma2=0.05):
		self.mode = mode
		self.learning_rate = learning_rate
		self.fx_dim = fx_dim	# dimension of noise vector used for generation
		self.beta = beta	# momentum term of optimizer
		# weight of regularization loss
		self.gamma1 = gamma1
		self.gamma2 = gamma2
		
	def generator(self, inputs, reuse=False):
		# inputs: [batch_size, 1, 1, 100]
		with tf.variable_scope('generator', reuse=reuse):
			with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None, stride=2, weights_initializer=tf.random_normal_initializer(stddev=0.02)):
				with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
					
					net = slim.conv2d_transpose(inputs, 512, [4, 4], padding='VALID', scope='conv_transpose1')
					net = slim.batch_norm(net, scope='bn1')		# [batch_size, 4, 4, 512]
					net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')
					net = slim.batch_norm(net, scope='bn2')		# [batch_size, 8, 8, 256]
					net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')
					net = slim.batch_norm(net, scope='bn3')		# [batch_size, 16, 16, 128]
					net = slim.conv2d_transpose(net, 3, [3, 3], activation_fn=tf.nn.tanh, scope='conv_transpose4')
					# [batch_size, 32, 32, 3]
		return net
		
	def discriminator(self, images, reuse=False):
		# images: [batch_size, 32, 32, 3]
		with tf.variable_scope('discriminator', reuse=reuse):
			with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None, stride=2, weights_initializer=tf.random_normal_initializer(stddev=0.02)):
				with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, activation_fn=None, is_training=(self.mode=='train')):
					
					net = slim.conv2d(images, 128, [3, 3], scope='conv1')
					net = slim.batch_norm(net, scope='bn1')
					net = tf.nn.relu(net) + 0.2 * tf.nn.relu(-net)		# [batch_size, 16, 16, 128]
					net = slim.conv2d(net, 256, [3, 3], scope='conv2')
					net = slim.batch_norm(net, scope='bn2')
					net = tf.nn.relu(net) + 0.2 * tf.nn.relu(-net)		# [batch_size, 8, 8, 256]
					net = slim.conv2d(net, 512, [3, 3], scope='conv3')
					net = slim.batch_norm(net, scope='bn3')
					net = tf.nn.relu(net) + 0.2 * tf.nn.relu(-net)		# [batch_size, 4, 4, 512]
					net = slim.conv2d(net, 1, [4, 4], padding='VALID', scope='conv4')
					net = slim.flatten(net)		# [batch_size, 1]
		return net
		
	def build_model(self):
		
		if self.mode == 'eval':
			# generate fake images with trained generator
			self.fx = tf.placeholder(tf.float32, [None, 1, 1, self.fx_dim], 'noise')
			self.sampled_images = self.generator(self.fx)
			# judge generated images with trained discriminator
			self.f_logits = self.discriminator(self.sampled_images)
			pass_rate = tf.round(tf.nn.sigmoid(self.f_logits))
			self.pass_rate = tf.reduce_mean(pass_rate)
			
		if self.mode == 'train':
			# generate fake images from noise vector and discriminate
			self.fx = tf.placeholder(tf.float32, [None, 1, 1, self.fx_dim], 'noise')
			self.fake_images = self.generator(self.fx)
			self.f_logits = self.discriminator(self.fake_images)
			
			# optimizer
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta)
			
			all_vars = tf.trainable_variables()
			d_vars = [var for var in all_vars if 'discriminator' in var.name]
			g_vars = [var for var in all_vars if 'generator' in var.name]
			
			# generator loss
			g_regularizer = tf.add_n([tf.nn.l2_loss(var) for var in g_vars])
			self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.f_logits), logits=self.f_logits) + self.gamma1 * g_regularizer
			self.g_loss = tf.reduce_mean(self.g_loss)
			g_loss_summary = tf.summary.scalar('generator_loss', self.g_loss)
					
			# g_train_op
			self.g_train_op = slim.learning.create_train_op(self.g_loss, self.optimizer, variables_to_train=g_vars)
			
			# attain real images from dataset and discriminate
			self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'cifar_images')
			self.logits = self.discriminator(self.images, reuse=True)
			
			# discriminator loss
			d_regularizer = tf.add_n([tf.nn.l2_loss(var) for var in d_vars])
			d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.logits), logits=self.logits)
			d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.f_logits), logits=self.f_logits)
			self.d_loss = d_loss_real + d_loss_fake + self.gamma2 * d_regularizer
			self.d_loss = tf.reduce_mean(self.d_loss)
			d_loss_summary = tf.summary.scalar('discriminator_loss', self.d_loss)
			
			self.summary = tf.summary.merge([d_loss_summary, g_loss_summary])
			
			# d_train_op
			self.d_train_op = slim.learning.create_train_op(self.d_loss, self.optimizer, variables_to_train=d_vars)
			
			for var in tf.trainable_variables():
				tf.summary.histogram(var.op.name, var)
