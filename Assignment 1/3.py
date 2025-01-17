price = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]

ask_product = int(input("Please select product (1-10) 0 to Quit:"))

while ask_product != 0
    print(f"Product: {ask_product} Price : {price[ask_product - 1]}")