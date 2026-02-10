# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.
<img width="1135" height="914" alt="Screenshot 2026-02-10 112708" src="https://github.com/user-attachments/assets/22132530-7efe-4d75-82e5-0ea4b31c84d9" />

## DESIGN STEPS
### STEP 1: 
Data Loading and Initial Inspection: The customers.csv dataset was loaded into a pandas DataFrame, and its initial structure was reviewed.

### STEP 2: 
Data Preprocessing: Irrelevant 'ID' column was dropped, missing values were filled (Work_Experience with 0, Family_Size with median), and all categorical features and the target 'Segmentation' were encoded into numerical representations using LabelEncoder.



### STEP 3: 
Data Splitting and Scaling: The dataset was split into training and testing sets (80/20 ratio), and numerical features were standardized using StandardScaler to ensure uniform scaling.



### STEP 4: 
Data Preparation for PyTorch: The processed numerical data was converted into PyTorch tensors and organized into TensorDataset and DataLoader objects for efficient batch processing during model training.



### STEP 5: 
Neural Network Definition, Training, and Evaluation: A PeopleClassifier neural network was defined, initialized with CrossEntropyLoss and Adam optimizer, trained for 100 epochs, and then evaluated on the test set to calculate accuracy, confusion matrix, and a classification report.


### STEP 6: 

Prediction and Visualization: The trained model was used to predict the segmentation for a sample input, and a heatmap of the confusion matrix was generated to visualize the model's performance.



## PROGRAM

### Name: Sanchita Sandeep

### Register Number: 212224240142

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

        




    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs,labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()





    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


model = PeopleClassifier(X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name: Sanchita Sandeep         ")
print("Register No:   212224240142    ")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name:  Sanchita Sandeep        ")
print("Register No:   212224240142    ")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')



```

### Dataset Information

<img width="912" height="319" alt="Screenshot 2026-02-10 112051" src="https://github.com/user-attachments/assets/4a345baa-e645-43aa-8dfc-1baa08944680" />

### OUTPUT

## Confusion Matrix


<img width="770" height="560" alt="Screenshot 2026-02-10 111213" src="https://github.com/user-attachments/assets/3a0df100-0054-4743-8203-f15e1a4830fe" />

## Classification Report

<img width="669" height="435" alt="Screenshot 2026-02-10 111223" src="https://github.com/user-attachments/assets/64adea64-0ab0-4203-a541-c706568a41dd" />

### New Sample Data Prediction

<img width="364" height="87" alt="Screenshot 2026-02-10 111057" src="https://github.com/user-attachments/assets/60091a8e-cccf-4786-8bbb-ee97379537a6" />

## RESULT
The program has been executed successfully
