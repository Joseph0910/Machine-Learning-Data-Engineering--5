class NeuralNetwork:

	# Attributes #
	def __init__(self,period_num,steps,train_x,train_y,test_x,test_y,hidden_units,batch_size,
				my_optimizer,test_data_x,prediction_data,nn_model,regularization_strength,learning_rate):
		self.period_num = period_num
		self.steps = steps
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.hidden_units = hidden_units
		self.batch_size = batch_size
		self.my_optimizer = my_optimizer
		self.prediction_data = prediction_data
		self.nn_model = nn_model
		self.regularization_strength = regularization_strength
		self.learning_rate = learning_rate

	def model_construct(self):
  		periods = period_num
 		steps_per_period = steps / periods
	  
		# Create a DNNRegressor object.
		# my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
		# my_optimizer=tf.train.AdagradOptimizer(learning_rate= learning_rate,l1_regularization_strength = regularization_strength)
		# my_optimizer=tf.train.AdamOptimizer(learning_rate= learning_rate, l1_regularization_strength = regularization_strength)	
		my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate, 
														l1_regularization_strength=regularization_strength)
		dnn_regressor = tf.estimator.DNNRegressor(
		    feature_columns=construct_feature_columns(train_x),
		    hidden_units=hidden_units,
		    optimizer=my_optimizer
		)
  
		# Create input functions # 
		training_input_fn = lambda: my_input_fn(train_x, 
		                                        train_y["target_field"], 
		                                        batch_size=batch_size)
		predict_training_input_fn = lambda: my_input_fn(train_x, 
		                                                train_y["target_field"], 
		                                                num_epochs=1, 
		                                                shuffle=False)
		predict_validation_input_fn = lambda: my_input_fn(test_x, 
		                                                test_y["target_field"], 
		                                                num_epochs=1, 
		                                                shuffle=False)

	def optimization(self):
		print("Training model...")
		print("RMSE (on training data):")

		training_rmse = []
		validation_rmse = []
  			
  			for period in range (0, periods): # Train the model, starting from the prior state.
				dnn_regressor.train( input_fn=training_input_fn, steps=steps_per_period )

			    # Take a break and compute predictions.
			    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
			    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
			    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
			    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
			    # Compute training and validation loss.
			    training_root_mean_squared_error = math.sqrt( 
			    	metrics.mean_squared_error(training_predictions, training_targets))
			    validation_root_mean_squared_error = math.sqrt(
			        metrics.mean_squared_error(validation_predictions, validation_targets))

			    # Occasionally print the current loss.
			    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
			    
			    # Add the loss metrics from this period to our list.
			    training_rmse.append(training_root_mean_squared_error)
			    validation_rmse.append(validation_root_mean_squared_error)
			
			print("Model training finished.")

			# Output a graph of loss metrics over periods.
			plt.ylabel("RMSE")
			plt.xlabel("Periods")
			plt.title("Root Mean Squared Error vs. Periods")
			plt.tight_layout()
			plt.plot(training_rmse, label="training")
			plt.plot(validation_rmse, label="validation")
			plt.legend()

			print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
			print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  		return dnn_regressor, training_rmse, validation_rmse

  	def predict():

		test_predictions = dnn_regressor.predict(input_fn= test_x)
		test_predictions = np.array([item['predictions'][0] for item in test_predictions])

		root_mean_squared_error = math.sqrt(
    		metrics.mean_squared_error(test_predictions, test_y))

		print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

	def predict_variance(self):

		ave_pred_val = mean(validation_predictions)
		ave_pred_test = mean(test_predictions)

		variance = abs(ave_pred_val - ave_pred_test)

		return variance

	def predict_bias(self):

		ave_predictions = mean(prediction_data)
		ave_actuals = mean(test_y)

		emperical_bias = abs(ave_predictions - ave_actuals)

		return emperical_bias


	def calibration_plot_bias(self):

		plt.figure(figsize=(10, 10))
		ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
		ax2 = plt.subplot2grid((3, 1), (2, 0))

		ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
		for clf, name in [(nn_model, 'Neural Network')]:
		    clf.fit(train_x, train_y)
		    if hasattr(clf, "predict_proba"):
		        prob_pos = clf.predict_proba(X_test)[:, 1]
		    else:  # use decision function
		        prob_pos = clf.decision_function(X_test)
		        prob_pos = \
		            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
		    fraction_of_positives, mean_predicted_value = \
		        calibration_curve(y_test, prob_pos, n_bins=10)

		    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
		             label="%s" % (name, ))

		    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
		             histtype="step", lw=2)

		ax1.set_ylabel("Fraction of positives")
		ax1.set_ylim([-0.05, 1.05])
		ax1.legend(loc="lower right")
		ax1.set_title('Calibration plots  (reliability curve)')

		ax2.set_xlabel("Mean predicted value")
		ax2.set_ylabel("Count")
		ax2.legend(loc="upper center", ncol=2)

		plt.tight_layout()
		plt.show()