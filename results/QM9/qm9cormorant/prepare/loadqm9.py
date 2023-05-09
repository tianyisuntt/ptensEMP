from numpy import load

test = load('./data/qm9/test.npz')
print(test)
print("===")
lst = test.files
print(lst)
for item in lst:
    print(item)
    print(test[item])

train = load('./data/qm9/train.npz')
print(train)
print("===")
lst = train.files
print(lst)
for item in lst:
    print(item)
    print(train[item])

valid = load('./data/qm9/valid.npz')
print(valid)
print("===")
lst = valid.files
print(lst)
for item in lst:
    print(item)
    print(valid[item])


# 22
# 19
