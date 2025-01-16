def tester(givenstring="Too short"):
    # Print the default value if length of string is < 10
    if len(givenstring.strip()) < 10:
        print(tester.__defaults__)
    else:
        print (givenstring)

def main():
    while True:
        prompt = input("Write something (quit ends): ")
        if prompt.lower() == "quit":
            break
        # Calling tester function
        tester(prompt)

#Run main function
main()
