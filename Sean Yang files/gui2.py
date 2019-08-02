file = open('gui.txt','r')
text=""
for line in file:
	text+=line[3:];
print(text)	
