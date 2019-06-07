#Imitation learning
"""
Example usage:
	python imitation_learning.py expert_data/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 200

"""

# Expert policy load
import tensorflow as tf
import numpy as np 
import gym
import argparse
import pickle

def build_model():

	xs = tf.placeholder(tf.float32, [None, 376])
	ys = tf.placeholder(tf.float32, [None, 17])
	out = tf.placeholder(dtype=tf.float32, shape=[None, 17])
	
	out = tf.layers.dense(xs, 256, activation = tf.nn.relu)
	out = tf.layers.dense(out, 300, activation = tf.nn.relu)
	out = tf.layers.dense(out, 196, activation = tf.nn.relu)
	out = tf.layers.dense(out, 98, activation = tf.nn.relu)
	out = tf.layers.dense(out, 32, activation = tf.nn.relu)
	out = tf.layers.dense(out, 17)
	return xs,ys,out


def main():
	sess = tf.Session()
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=200,
						help='Number of expert roll outs')
	args = parser.parse_args()

	print('loading expert data')
	filename = args.expert_policy_file

	with open(filename, 'rb') as f:
		data = pickle.loads(f.read())

	# assert len(data.keys()) == 2
	observations = data['observations']
	actions = data['actions']

	observations = np.squeeze(observations)
	actions = np.squeeze(actions)

	input_ph, output_ph, output_pred = build_model()
	
	# create loss
	mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

	# create optimizer
	opt = tf.train.AdamOptimizer(learning_rate=0.03).minimize(mse)

	# initialize variables
	sess.run(tf.global_variables_initializer())
	# create saver to save model variables
	saver = tf.train.Saver()

	# run training
	batch_size = 32
	for training_step in range(10000):
		# get a random subset of the training data
		
		# run the optimizer and get the mse
		_,output_pred_run, mse_run = sess.run([opt,output_pred, mse], feed_dict={input_ph: observations, output_ph: actions})
		if training_step % 100 == 0:
			print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
			print((output_pred_run - actions).mean())
			print((output_pred_run - actions).sum())
		# print the mse every so often
		if training_step % 200 == 0:
			saver.save(sess, "store/model.ckpt")

def test():
	tf.reset_default_graph()
	sess = tf.Session()
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=200,
						help='Number of expert roll outs')
	args = parser.parse_args()

	print('loading expert')
	filename = args.expert_policy_file

	with open(filename, 'rb') as f:
		data = pickle.loads(f.read())

	# assert len(data.keys()) == 2
	observations = data['observations']
	actions = data['actions']

	observations = np.squeeze(observations)
	actions = np.squeeze(actions)

	# create the model
	input_ph, output_ph, output_pred = build_model()

	# restore the saved model
	saver = tf.train.Saver()
	saver.restore(sess, "store/model.ckpt")

	output_pred_run = sess.run(output_pred, feed_dict={input_ph: observations})

	#plt.scatter(inputs[:, 0], outputs[:, 0], c='k', marker='o', s=0.1)
	#plt.scatter(inputs[:, 0], output_pred_run[:, 0], c='r', marker='o', s=0.1)

	print((output_pred_run - actions).sum())

def gymtest():

	tf.reset_default_graph()
	sess = tf.Session()
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=200,
						help='Number of expert roll outs')
	args = parser.parse_args()

	print('loading model')

	# create the model
	input_ph, output_ph, output_pred = build_model()

	# restore the saved model
	saver = tf.train.Saver()
	saver.restore(sess, "store/model.ckpt")

	with tf.Session():
		#tf_util.initialize()

		env = gym.make(args.envname)
		max_steps = args.max_timesteps or env.spec.timestep_limit
		returns = []
		observations = []
		actions = []
		for i in range(args.num_rollouts):
			print('iter', i)
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				action = sess.run(output_pred, feed_dict={input_ph: obs[None,:]})
				
				#print(action)
				observations.append(obs)
				actions.append(action)
				obs, r, done, _ = env.step(action)
				totalr += r
				steps += 1
				if args.render:
					env.render()
				if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
				if steps >= max_steps:
					break
			returns.append(totalr)

		print('returns', returns)
		print('mean return', np.mean(returns))
		print('std of return', np.std(returns))




if __name__ == '__main__':
	main()
	test()
	#gymtest()
