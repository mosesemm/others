


filename = "./work_data.txt"

'''
file_object = open(filename, mode="r+")

print(file_object.read())

if file_object:
    file_object.close()
    
'''


#Another way to do it:

with open(filename, "r+") as fileData:
    #line = fileData.read(6)
    #line = fileData.readline()
    #print(len(line))
    #print(line)

    for line in fileData:
        print(line)


#writing to file
with open(filename, "a+") as fileData:
    fileData.write("Last line to be added\n")
    fileData.write(str([13,44,66])+"\n")
    print(fileData.tell())
    fileData.seek(0,0)
    print(fileData.readline())

work_data_contents = []
with open(filename, "r") as fileData:
    work_data_contents = fileData.readlines()

work_data_contents.insert(1, "This goes between line 1 and 2\n")

with open(filename, "w") as fileData:
    fileData.write("".join(work_data_contents))


print("file probably closed: {} ".format(filename))