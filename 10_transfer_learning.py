# Transfer Learning #

model = models.vgg16(pretrained = True) # weights and biases optimized # 

# New Classifier to replace vg16 #

classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,512))
							('relu',nn.ReLU()),
							('dropout',nn.Dropout(p=0.337)),
							('fc2',nn.Linear(512,102)),
							('output',nn.LogSoftmax(dim=1))
							]))

# Transfer the classfier over #
model.classifier = classifier

# Define criteria and optimizer # 
criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.005,momentum = 0.5)
# momentum: amt by which a model can bump off any local minimum during GD #

# Training Function #

def train(model,loader,criterion,gpu):
	model.train() 
	current_loss = 0 
	current_correct = 0

	for train,y_train in iter(loader):
		if gpu:
			train,y_train = train.to('cuda',y_train.to('cuda'))
		optimizer.zero_grad()
		output = model.forward(train)
		_,preds = torch.max(output,1)
		loss = criterion(output,y_train)
		loss.backward()
		optimizer.step()
		current_loss += loss.item()*train_size(0)
		current_correct += torch.sum(preds == y_train.data)
	epoch_loss = current_loss / len(trainLoader.dataset)
	epoch_acc = current_correct.double() / len(trainLoader.dataset)

	return epoch_loss, epoch_acc

# Validation Function #

def validation(model,loader,criterion,gpu):
	model.eval()
	valid_loss = 0
	valid_correct = 0
	for valid,y_valid in iter(loader):
		if gpu:
			valid,y_valid = valid.to('cuda',y_valid.to('cuda'))
		output = model.forward(valid)
		valid_loss += criterion(output,y_valid).item()*valid.size(0)
		equal = (output.max(dim=1)[1] == y_valid.data)
		valid_correct += torch.sum(equal)#type(torch.FloatTensor)

	epoch_loss = valid_loss / len(validLoader.dataset)
	epoch_acc = valid_correct.double() / len(validLoader.dataset)

	return epoch_loss, epoch_acc

# Combine both function #

for param in model.parameters():
	param.require_grad = False

epochs = 10 
epoch = 0

# send model to GPU #
if args.gpu:
	model.to('cuda')

for e in range(epochs):
	epoch += 1
	print(epoch)
	with torch.set_grad_enabled(True):
		epoch_train_loss, epoch_train_acc = train(model,trainLoader,criteria,args.gpu)
	print("Epoch: {} Train Loss: {:4f} Train Accuracy: {:4f}".format(epoch,epoch_train_loss,epoch_train_acc))

with torch.no_grad():
	epoch_val_loss, epoch_val_acc = valiation(model,validLoader,criteria,args.gpu)
	print("Epoch: {} Valid Loss: {:4f} Valid Accuracy: {:4f}".format(epoch,epoch_val_loss,epoch_val_acc))

# Test Model #

model.eval()
total = 0 
count = 0 

for test,y_test in iter(testLoader):
	test, y_test = test.to('cuda'),y_test.to('cuda')

# Calculate class probabilities (softmax) for img # 

with torch.no_grad():
	output = model.forward(test)
	ps = torch.exp(output)
	_,predicted = torch.max(output.data,1)
	total += y_test.size(0)
	correct += (predicted == y_test).sum().item()
	count += 1 
	print("Accuracy of network on test images is ... {:4f}...count:{}".format(100*correct/total,count))

# Save model #

# create checkpoint and save every sensitive info starting from model state dictionary
# model criterion, optimizer, to the # number of epochs 

checkpoint = {'model_state': model.state_dict(),
			'criterion_state': criteria.state_dict(),
			'optimizer_state': optimizer.state_dict(),
			'class_to_idx': train_datasets.class_to_idx,
			'epochs': epochs,
			'Best train loss': epoch_train_loss,
			'Best train accuracy': epoch_train_accuracy,
			'Best validation loss': epoch_val_loss,
			'Best validation accuracy' : epoch_val_acc}

torch.save(checkpoint,args.checkpoint)





