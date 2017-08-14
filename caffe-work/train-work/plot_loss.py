import matplotlib.pyplot as plt
import numpy as np
loss_file = "loss_values.txt"
with open(loss_file, 'r') as f:
    lines = f.readlines()
loss = np.zeros(len(lines))
print(len(lines))
x = np.linspace(1, len(lines), len(lines))
x = x * 10000
print(x)
for i,line in enumerate(lines):
    loss[i] = float(line.strip().split("'")[0])

print(loss)
plt.figure(figsize=(8,4)) 
plt.plot(x,loss,label="$train_loss$",color="red",linewidth=2)  
plt.xlabel("iterations")  
plt.ylabel("loss")  
plt.title("loss plot")  
plt.ylim(0,10)  
plt.show()  

