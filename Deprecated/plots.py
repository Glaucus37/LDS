import matplotlib.pyplot as plt

print("Hello!")
f1 = open('energy1.txt', 'r')
f2 = open('energy2.txt', 'r')
# energy = f.read().splitlines()
energy1 = []
energy2 = []

for line in f1:
    energy1.append(float(line))

for line in f2:
    energy2.append(float(line))

if energy1[-1] == 0.:
    energy1.pop()

if energy2[-1] == 0.:
    energy2.pop()

x = range(len(energy1))

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, energy1)
ax.plot(x, energy2)
# plt.ylim()
# print(plt.xlim(), plt.ylim())

plt.show()
