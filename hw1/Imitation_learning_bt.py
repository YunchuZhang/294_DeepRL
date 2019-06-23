#Imitation learning
"""
Example usage:
	python imitation_learning.py expert_data/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 200


	Dagger
	python imitation_learning.py expert_data/Humanoid-v2.pkl experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 200 --Dagger_iter 10
"""

# Expert policy load
import tensorflow as tf
import numpy as np 
import gym
import argparse
import pickle
import load_policy

def build_model(batch):
	print(batch)
	xs = tf.placeholder(tf.float32, [None, 376])
	ys = tf.placeholder(tf.float32, [None, 17])
	out = tf.placeholder(dtype=tf.float32, shape=[None, 17])
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256, forget_bias=1.0, state_is_tuple=True)
	init_state = lstm_cell.zero_state(batch,dtype=tf.float32) 
#dagger batch = 1 other 128
	out = tf.reshape(xs,[batch,1,376])
	print(out.shape)
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, out, initial_state=init_state, time_major=False)
	print(outputs.shape)
	outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
	#print(len(outputs))
	print(outputs[-1].shape)
	
	out = tf.layers.dense(outputs[-1], 196, activation = tf.nn.relu)
	print(out.shape)
	out = tf.layers.dense(out, 128, activation = tf.nn.relu)
	out = tf.layers.dense(out, 64, activation = tf.nn.relu)
	out = tf.layers.dense(out, 32, activation = tf.nn.relu)



	out = tf.layers.dense(out, 17)
	print(out.shape)
	return xs,ys,out


def main():
	sess = tf.Session()
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_data', type=str)
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=200,
						help='Number of expert roll outs')
	parser.add_argument('--Dagger_iter', type = int, default=10)
	args = parser.parse_args()

	print('loading expert data')
	filename = args.expert_policy_data

	with open(filename, 'rb') as f:
		data = pickle.loads(f.read())

	# assert len(data.keys()) == 2
	observations = data['observations']
	actions = data['actions']

	observations = np.squeeze(observations)
	actions = np.squeeze(actions)

	input_ph, output_ph, output_pred = build_model(batch = 128)
	
	# create loss
	mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

	# create optimizer
	opt = tf.train.AdamOptimizer(learning_rate=0.000006).minimize(mse)

	# initialize variables
	sess.run(tf.global_variables_initializer())
	# create saver to save model variables
	saver = tf.train.Saver()




# run training
	batch_size = 128

	batch_idxs = observations.shape[0] // batch_size


	for training_step in range(20000):
		# get a random subset of the training data
		
		for idx in range(batch_idxs):

			batch_train = observations[idx * batch_size : (idx + 1) * batch_size]
			batch_value = actions[idx * batch_size : (idx + 1) * batch_size]

		
			# run the optimizer and get the mse
			_,output_pred_run, mse_run = sess.run([opt,output_pred, mse], feed_dict={input_ph: batch_train, output_ph: batch_value})
			if training_step%100 == 0 and idx % 50 == 0:
				print('Epoch:{0:04d}'.format(training_step))
				print('Batch:{0:04d}mse: {1:.6f}'.format(idx,mse_run))
				print((output_pred_run - batch_value).mean())
				print((output_pred_run - batch_value).sum())
		# print the mse every so often
		if training_step % 200 == 0:
			saver.save(sess, "store/model.ckpt")

def Dagger():

	tf.reset_default_graph()
	sess = tf.Session()
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_data', type=str)
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=20,
						help='Number of expert roll outs')
	parser.add_argument('--Dagger_iter', type = int, default=10)
	args = parser.parse_args()

	print('loading model')

	# create the model
	input_ph, output_ph, output_pred = build_model(batch = 1)

	# restore the saved model
	saver = tf.train.Saver()
	saver.restore(sess, "store/model.ckpt")


	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)

	print('loaded and built')

	with tf.Session():
		#tf_util.initialize()

		env = gym.make(args.envname)
		max_steps = args.max_timesteps or env.spec.timestep_limit
		returns = []
		observations = []
		actions = []
		for i in range(args.Dagger_iter):
			print('iter', i)
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				action = sess.run(output_pred, feed_dict={input_ph: obs[None,:]})
				real_action = policy_fn(obs[None,:])
				
				#print(action)
				observations.append(obs)
				actions.append(real_action)
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


		observations_n = np.array(observations)
		actions_n = np.array(actions)



	#Train
	tf.reset_default_graph()
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	print('loading expert data')
	filename = args.expert_policy_data

	with open(filename, 'rb') as f:
		data = pickle.loads(f.read())

	# assert len(data.keys()) == 2
	observations = data['observations']
	actions = data['actions']

	observations = np.squeeze(observations)
	actions = np.squeeze(actions)
	actions_n = np.squeeze(actions_n)

	#print(observations.shape)
	print(observations_n.shape)

	#print(actions.shape)
	print(actions_n.shape)

	observations = np.concatenate((observations, observations_n), axis = 0)
	print(observations.shape)
	actions = np.concatenate((actions, actions_n), axis = 0)
	
	input_ph, output_ph, output_pred = build_model(batch = 128)
	
	# create loss
	mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

	# create optimizer
	opt = tf.train.AdamOptimizer(learning_rate=0.000003).minimize(mse)

	# initialize variables
	sess.run(tf.global_variables_initializer())
	# create saver to save model variables
	saver = tf.train.Saver()


# run training
	batch_size = 128

	batch_idxs = observations.shape[0] // batch_size


	for training_step in range(20000):
		# get a random subset of the training data
		
		for idx in range(batch_idxs):

			batch_train = observations[idx * batch_size : (idx + 1) * batch_size]
			batch_value = actions[idx * batch_size : (idx + 1) * batch_size]

		
			# run the optimizer and get the mse
			_,output_pred_run, mse_run = sess.run([opt,output_pred, mse], feed_dict={input_ph: batch_train, output_ph: batch_value})
			if training_step%100 == 0 and idx % 50 == 0:
				print('Epoch:{0:04d}'.format(training_step))
				print('Batch:{0:04d}mse: {1:.6f}'.format(idx,mse_run))
				print((output_pred_run - batch_value).mean())
				print((output_pred_run - batch_value).sum())
		# print the mse every so often
		if training_step % 200 == 0:
			saver.save(sess, "store/model1.ckpt")





def test():
	tf.reset_default_graph()
	sess = tf.Session()
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_data', type=str)
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=100,
						help='Number of expert roll outs')
	parser.add_argument('--Dagger_iter', type = int, default=10)
	args = parser.parse_args()


	print('loading expert')
	filename = args.expert_policy_data

	with open(filename, 'rb') as f:
		data = pickle.loads(f.read())

	# assert len(data.keys()) == 2
	observations = data['observations']
	actions = data['actions']

	observations = np.squeeze(observations)
	actions = np.squeeze(actions)

	# create the model
	input_ph, output_ph, output_pred = build_model(batch = 1)

	# restore the saved model
	saver = tf.train.Saver()
	saver.restore(sess, "store/model.ckpt")

	batch_size = 128

	batch_idxs = observations.shape[0] // batch_size


		
	for idx in range(batch_idxs):

		batch_train = observations[idx * batch_size : (idx + 1) * batch_size]
		batch_value = actions[idx * batch_size : (idx + 1) * batch_size]

	
		# run the optimizer and get the mse
		output_pred_run = sess.run(output_pred, feed_dict={input_ph: batch_train, output_ph: batch_value})
		if idx % 30 == 0:

			print('Batch:{0:04d}'.format(idx))
			print((output_pred_run - batch_value).mean())
			print((output_pred_run - batch_value).sum())



def gymtest():

	tf.reset_default_graph()
	sess = tf.Session()
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_data', type=str)
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=100,
						help='Number of expert roll outs')
	parser.add_argument('--Dagger_iter', type = int, default=10)
	args = parser.parse_args()

	print('loading model')

	# create the model
	input_ph, output_ph, output_pred = build_model(batch = 1)

	# restore the saved model
	saver = tf.train.Saver()
	saver.restore(sess, "store/model1.ckpt")

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
	#main()
	#test()
	#Dagger()
	gymtest()
