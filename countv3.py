import numpy as np 
import matplotlib.pyplot as plt
v = []
for z in range(6): 
 os=np.loadtxt("op"+str(z)+".csv",delimiter=',')[:,1]
 v.append([])
 for i in range(5):
      q=0
      
      for j in range(400):
        if os[j]==i:
          q=q+1
      v[z].append(q)
print np.transpose(np.array(v))
 

f = []
f = np.transpose(np.array(v))
f = np.array(f)
plt.plot(f)
plt.show()
fd = file('resultn3.csv','a')
np.savetxt(fd,f,delimiter=",")
fd.close()



 
           
 
