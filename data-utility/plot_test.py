from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = sns.load_dataset('iris')


# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)
# def g(t):
#     return np.sin(t) * np.cos(1/(t+0.1))

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)

# fig = plt.figure(1, figsize=(10,4))
# fig.suptitle('First figure instance')


# plt.figure(2, figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.bar(t1, f(t1))
# plt.subplot(1, 2, 2)
# plt.scatter(t2, g(t2))
# plt.suptitle('second plot')
# plt.show(block=False)
# input('...')

# sub1 = fig.add_subplot(221)
# sub1.plot(t1, g(t1))
# sub2 = fig.add_subplot(224)
# sub2.plot(t2, f(t2))
# plt.show(block=False)


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
def g(t):
    return np.sin(t) * np.cos(1/(t+0.1))

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)




# sub1 = fig.add_subplot(221)
# sub1.plot(t1, g(t1))
# sub2 = fig.add_subplot(224)
# sub2.plot(t2, f(t2))
# plt.show(block=False)
# fig.savefig("first_figure.png", dpi=200)
# fig.savefig("first_figure.png", dpi=200)

f, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
sns.distplot( df["sepal_length"] , color="skyblue", ax=axes[0, 0])
sns.distplot( df["sepal_width"] , color="olive", ax=axes[0, 1])
sns.distplot( df["petal_length"] , color="gold", ax=axes[1, 0])
sns.distplot( df["petal_width"] , color="teal", ax=axes[1, 1])
plt.show(block=False)



plt.figure(2, figsize=(10,5))
plt.scatter(t2, g(t2))
plt.suptitle('first plot')
plt.show(block=False)
input('...')



sns.distplot( df["sepal_length"] , color="gold", ax=axes[0, 0])
sns.distplot( df["sepal_width"] , color="gold", ax=axes[0, 1])
sns.distplot( df["petal_length"] , color="gold", ax=axes[1, 0])
sns.distplot( df["petal_width"] , color="gold", ax=axes[1, 1])
plt.show()


