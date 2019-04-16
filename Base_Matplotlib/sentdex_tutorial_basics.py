import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

x = [1,2,3]
y = [5,7,3]

plt.plot(x,y)
plt.show()


#Legends, Titles and Labels 

x2 = [1,2,3]
y2 = [10,14,12]

plt.plot(x,y, label = 'First Line', linewidth = 5)
plt.plot(x2, y2, label = 'Second Line', linewidth = 5)
plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.grid(True, color = 'k')
plt.show()


#Bar chart 
x = [2,4,6,8,10]
y = [4,7,5,4,6]

x2 = [1,3,5,7,9]
y2 = [3,7,8,4,2]

plt.bar(x,y, label = 'Bars1', color = 'k')
plt.bar(x2, y2, label = 'Bars2', color = 'c')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show

#Histogram 
pop_ages = [22,32,63,45,64,75,2,32,123,12,111,12,123,43,56,76,54,23,24,54,67,89,
            121,123,4,66,78,4,5,6,98]
bins = [x for x in range(0,130, 10)]
plt.hist(pop_ages, bins, histtype = 'bar', rwidth = 0.8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.show

#scatterplot
x = [1,2,3,4,5,6,7,8]
y = [4,2,6,3,6,8,4,3]

x2 = [1,2,3,4,5,6,7,8]
y2 = [2,5,7,1,2,5,3,7]


plt.scatter(x,y, label = 'skitscat', color = 'k', s = 100)
plt.scatter(x2,y2, label = 'scatskit', color = 'c', marker = '*', s = 100)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show



