import numpy as np

# a = np.array([[10,20,30],[2,4,5]])
# print(a.shape)
# print(a.dtype)


# zeroes = np.zeros((3,4))
# print(zeroes)
# ones = np.ones((2,4))
# print(ones)

# rng = np.arange(0,10,3)
# print(rng)

# rand = np.random.randn(1,1)
# print(rand)

# indexing & slicing

data = np.array ([[1,2,3,4] ,[5,6,7,8] , [9,10,11,12]])
# print(data[0])
# print(data[0,2])
# print(data[:,0])
# print(data[:,:2])

# stats
avg = data.mean(axis=1)
std = data.std(axis=1)
max = data.max(axis=1)
print(f"here is avg {avg}")
print(f"here is std {std}")
print(f"here is max {max}")

#  Dot product 
features = np.array([120,46,2,30])
weight = np.array([2000,3000,4000,5000])
price = np.dot(features,weight)

print(price)

# Normalizatio
# TASK: You have exam scores for 5 students across 3 subjects
scores = np.array([
    [85, 92, 78],   # student 0: math, science, english
    [90, 88, 95],   # student 1
    [70, 65, 80],   # student 2
    [95, 97, 92],   # student 3
    [60, 72, 68],   # student 4
])
avg_eachStudent = scores.mean(axis=1)
avg_eachSubj = scores.mean(axis=0)
top = avg_eachStudent.max(axis=1)
m1 = scores.mean(axis =0)
std1 = scores.std(axis=1)
norm= (data-m1)/std1


# Answer these using NumPy — no for-loops allowed:
# 1. What is each student's average score?   → shape should be (5,)
# 2. What is the class average per subject?  → shape should be (3,)
# 3. Who is the top student (highest average)?  → one index number
# 4. Normalize the scores (mean=0, std=1)
# 5. Which student scored above 85 in ALL subjects?  → boolean indexing

# Hint for Q5: boolean indexing looks like this:
# mask = (scores > 85).all(axis=1)
# print(scores[mask])