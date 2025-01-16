shopping_list = []

while True:
    action = input("Would you like to \n (1) Add or \n (2) Remove items \n (3) Quit \n").strip()
    if action == "1":
        item_add = input("What will be added?")
        shopping_list.append(item_add)
    elif action == "2":
        print(f"There are {len(shopping_list)} items in the list.")
        item_remove = int(input("Which item is deleted?"))
        shopping_list.pop(item_remove)
    elif action == "3":
        print(f"The following items remain in the list: ")
        print("\n".join(shopping_list))
        break
    else:
        print("Incorrect Selection")