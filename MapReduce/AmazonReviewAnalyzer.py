import csv
import io
data = []
with open('Amazon Reviews.csv', newline='') as csvfile:
    amazon_reader = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
    # print(amazon_reader[1][11])
    # amazon_reader_slice = [x for x in amazon_reader ]
    i=1
    for row in amazon_reader:
        if 55 < i < 60:
            print("----")
            print(row[11].replace("\n","").replace("\t",""))
        if len(row) > 0 and row[11].strip() != '':
            data.append(row[11].strip().replace("\n","")+ "\n")
        i = i + 1
    #     for item in row:
    #         print(item)

# f = open("Amazon Reviews.csv", "r")
#
# print(f.readline())
# line = f.readline()
# output = io.StringIO()
# writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
# print(writer.writerow(f.readline()))
# print(output.getvalue().rstrip('\r\n'))
# print(writer.writerow(f.readline()))
# print(line)
# print(line.split(',')[11])
# for split in line.split(','):
#     print(split)
#
# f.close()

# for x in f:
#   print(x)
with open("Amazon_Reviews_Clean.txt", "w") as f:
    # f.write("Created using write mode.")
    f.writelines(data)
    # for row in data:
    #     print(row)
    #     f.write(row)
# with open('Amazon_Reviews_Clean.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter='|')
#     for row in data:
#         print(row)
#         writer.writerow(row)