print("Enter Text : ")
string=input()
text=string.lower()
def vowel(letter):
    if (letter in ["a","e","i","o","u"]):
        return True
    else:
        return False
count1=0
count2=0
n=len(text)
for i in range(0,n):
    if (vowel(text[i])) :
        count1=count1+1
    elif (text[i] == " "):
        i=i+1
    else:
        count2=count2+1
print(count1)
print(count2)
