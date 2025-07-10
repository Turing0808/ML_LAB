import random

nums = []
for _ in range(100):
    nums.append(random.randint(0, 10))

# Mean
total = 0
for n in nums:
    total += n
avg = total / len(nums)

for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[j] < nums[i]:
            nums[i], nums[j] = nums[j], nums[i]

if len(nums) % 2 == 1:
    med = nums[len(nums) // 2]
else:
    mid1 = nums[len(nums) // 2 - 1]
    mid2 = nums[len(nums) // 2]
    med = (mid1 + mid2) / 2

max_freq = 0
mode = None
for i in range(len(nums)):
    count = 0
    for j in range(len(nums)):
        if nums[i] == nums[j]:
            count += 1
    if count > max_freq:
        max_freq = count
        mode = nums[i]

print("Mean:", avg)
print("Median:", med)
print("Mode:", mode)
